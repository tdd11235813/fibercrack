#ifndef CONFIGURATION_H
#define CONFIGURATION_H

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
  T zlevel_rel = 0.5;       // relative z-position of slice level
  T zlevel_abs = zlevel_rel * zmax;

  T NV=0;//0.0333;          // computed ... in 1/mm³
  T Vf=0.0333;          // in 1/mm³;
  T Af=0;          // ... computed
  T EL=0.0333;          // in 1/mm³;
  T fdiam=0.2;       // in mm
  T lambda = 0; //computed

  T mu=0.7;           //in rad^(-1)
  T sigmafu=2500.0;      // in MPA
  T d_end=0.3;                //in mm (changed from 3 to 0.3)
  uint n_d=200; //Anzahl der Schrittzahl bei der Spaltöffnung

//  T beta_c = 2.0;
  T weib_m = 10.0;
  T weib_xc = 10.0;

  T fis_k = 20.0;
};


#endif /* CONFIGURATION_H */
