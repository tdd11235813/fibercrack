#ifndef CONFIGURATION_H
#define CONFIGURATION_H

enum class PSingle_Model {
  Li,
  Pfyl
};

enum class Beta_Model {
  ACOS,
  RAND
};

/// Parameters used for kernels
template<typename T>
struct Configuration
{
  uint nr = 4; // number repetitions

  int rand_seed = 1337;
  int rand_offset = 0;

  T xmax=100.0;         //in mm
  T ymax=100.0;         //  ""
  T zmax=100.0;         //  ""
  T zlevel_rel = 0.5;   // relative z-position of slice level
  T zlevel_abs = zlevel_rel * zmax; // absolute z-position in mm

  T NV=0;               // ... computed
  T Vf=0.02;
  T Af=0;               // ... computed
  T EL=0.0333;
  T Ef=720;
  T Em=500;
  T nfm=Ef/Em;          // ... computed
  T nfm_Vf=nfm*Vf;      // ... computed
  T tau=1.02;
  T fdiam=0.2;          // in mm
  T lambda = 0;         // ... computed

  T mu=0.7;             // in rad^(-1)
  T sigma_fu=2500.0;    // in MPA
  T d_end=0.3;          // in mm (crack diameter)
  uint n_d=200;         // number of steps for crack diameter iteration

//  T beta_c = 2.0;
  T weib_m = 10.0;
  T weib_xc = 10.0;

  T fis_k = 20.0;
  T fis_b = 0.0;        // ... computed
  T fis_x0 = 0.0;       // ... computed
  T fis_c = 0.0;        // ... computed
  PSingle_Model psingle_model = PSingle_Model::Li;
  Beta_Model beta_model = Beta_Model::ACOS;
};


#endif /* CONFIGURATION_H */
