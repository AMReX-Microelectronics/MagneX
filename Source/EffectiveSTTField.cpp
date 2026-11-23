#include "MagneX.H"

using namespace amrex;

/*
  Spin-transfer torque (STT) from a current flowing perpendicular to the layers of a
  spin valve / magnetic tunnel junction (Slonczewski model).

  Both torques are written as effective fields, so that they are added to H_eff exactly
  like H_exchange, H_anisotropy, ... and work with every time integrator:

      dm/dt = -mu0 |gamma| m x ( H_eff + H_STT_DL + H_STT_TL ) + alpha m x dm/dt

      H_STT_DL = a_J (m x p)      damping-like (Slonczewski, "in-plane") torque
      H_STT_TL = b_J  p           field-like  ("perpendicular") torque

  where
      m        = M/|M|, unit vector of the local (free layer) magnetization
      p        = STT_polarization_axis, unit vector along the fixed (reference) layer
                 magnetization, i.e. the spin polarization of the current
      beta     = hbar J / (mu0 e |M| d)                                         [A/m]
      a_J      = beta * epsilon
      b_J      = beta * epsilon'
      epsilon  = P Lambda^2 / ( (Lambda^2 + 1) + (Lambda^2 - 1) (m . p) )
      J        = J_STT_parser(x,y,z,t), current density                         [A/m^2]
      d        = STT_free_layer_thickness                                       [m]
      P        = STT_polarization, Lambda = STT_Lambda, epsilon' = STT_epsilon_prime

  For Lambda = 1 this reduces to a_J = hbar J P / (2 mu0 e Ms d).  It is the same
  convention as OOMMF (Oxs_SpinXferEvolve) and MuMax3; in Landau-Lifshitz form

      dm/dt|STT = mu0 |gamma| / (1 + alpha^2) *
                  [ (a_J + alpha b_J) m x (p x m)  -  (b_J - alpha a_J) m x p ]

  Sign convention: J > 0 means electrons flow from the fixed layer into the free layer.
  Then a_J > 0 and the damping-like torque pulls m towards p (favors the parallel
  state); J < 0 favors the antiparallel state.

  The two terms are switched on/off independently with STT_DL_coupling and
  STT_TL_coupling.  The torque acts wherever J_STT_parser is non-zero, so the parser
  is also used to restrict the STT to the free layer.
*/
void CalculateH_STT(Array< MultiFab, AMREX_SPACEDIM> &   Mfield,
                    Array< MultiFab, AMREX_SPACEDIM> &   H_STT_DLfield,
                    Array< MultiFab, AMREX_SPACEDIM> &   H_STT_TLfield,
                    MultiFab&   Ms,
                    MultiFab&   J_STT)
{
    // timer for profiling
    BL_PROFILE_VAR("CalculateH_STT()",CalculateH_STT);

    // physical constants (SI)
    const Real hbar = 1.054571817e-34; // reduced Planck constant [J s]
    const Real q_e  = 1.602176634e-19; // elementary charge [C]

    // local copies of the run-time parameters, captured by value in the GPU kernel
    const int do_DL  = STT_DL_coupling;
    const int do_TL  = STT_TL_coupling;
    const int M_norm = M_normalization;

    const Real px = STT_polarization_axis[0];
    const Real py = STT_polarization_axis[1];
    const Real pz = STT_polarization_axis[2];

    const Real polarization  = STT_polarization;
    const Real Lambda2       = STT_Lambda * STT_Lambda;
    const Real epsilon_prime = STT_epsilon_prime;

    // beta = beta_coeff * J / |M|, with beta_coeff = hbar / (mu0 e d)
    const Real beta_coeff = (hbar / q_e) / (mu0 * STT_free_layer_thickness);

    for (MFIter mfi(Mfield[0], TilingIfNotGPU()); mfi.isValid(); ++mfi) {

        const Box& bx = mfi.tilebox();

        const Array4<Real const>& Mx = Mfield[0].const_array(mfi);
        const Array4<Real const>& My = Mfield[1].const_array(mfi);
        const Array4<Real const>& Mz = Mfield[2].const_array(mfi);
        const Array4<Real const>& Ms_arr = Ms.const_array(mfi);
        const Array4<Real const>& J_arr = J_STT.const_array(mfi);
        const Array4<Real>& Hx_STT_DL = H_STT_DLfield[0].array(mfi);
        const Array4<Real>& Hy_STT_DL = H_STT_DLfield[1].array(mfi);
        const Array4<Real>& Hz_STT_DL = H_STT_DLfield[2].array(mfi);
        const Array4<Real>& Hx_STT_TL = H_STT_TLfield[0].array(mfi);
        const Array4<Real>& Hy_STT_TL = H_STT_TLfield[1].array(mfi);
        const Array4<Real>& Hz_STT_TL = H_STT_TLfield[2].array(mfi);

        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                Real Hx_DL = 0., Hy_DL = 0., Hz_DL = 0.;
                Real Hx_TL = 0., Hy_TL = 0., Hz_TL = 0.;

                // determine if the material is magnetic or not
                if (Ms_arr(i,j,k) > 0.) {

                    // 0 = unsaturated; compute |M| locally.  1 = saturated; use M_s
                    const Real M_magnitude = (M_norm == 0) ? std::sqrt(Mx(i,j,k)*Mx(i,j,k) + My(i,j,k)*My(i,j,k) + Mz(i,j,k)*Mz(i,j,k))
                                                           : Ms_arr(i,j,k);

                    if (M_magnitude > 0.) {

                        // unit magnetization
                        const Real mx = Mx(i,j,k) / M_magnitude;
                        const Real my = My(i,j,k) / M_magnitude;
                        const Real mz = Mz(i,j,k) / M_magnitude;

                        // beta = hbar J / (mu0 e |M| d)  [A/m]
                        const Real beta = beta_coeff * J_arr(i,j,k) / M_magnitude;

                        if (do_DL == 1) {
                            // H_STT_DL = a_J (m x p), with the Slonczewski angular dependence in a_J
                            const Real m_dot_p = mx*px + my*py + mz*pz;
                            const Real epsilon = polarization * Lambda2 / ( (Lambda2 + 1.) + (Lambda2 - 1.) * m_dot_p );
                            const Real a_J = beta * epsilon;

                            Hx_DL = a_J * (my*pz - mz*py);
                            Hy_DL = a_J * (mz*px - mx*pz);
                            Hz_DL = a_J * (mx*py - my*px);
                        }

                        if (do_TL == 1) {
                            // H_STT_TL = b_J p
                            const Real b_J = beta * epsilon_prime;

                            Hx_TL = b_J * px;
                            Hy_TL = b_J * py;
                            Hz_TL = b_J * pz;
                        }
                    }
                }

                Hx_STT_DL(i,j,k) = Hx_DL;
                Hy_STT_DL(i,j,k) = Hy_DL;
                Hz_STT_DL(i,j,k) = Hz_DL;

                Hx_STT_TL(i,j,k) = Hx_TL;
                Hy_STT_TL(i,j,k) = Hy_TL;
                Hz_STT_TL(i,j,k) = Hz_TL;
            });
    }
}
