#ifndef CONFIGURATION_H
#define CONFIGURATION_H

/// Parameters used for kernels
template<typename T>
struct Configuration
{
  T lambda = 11e+7; // will be xmax*ymax*zmax*NV
  uint nr = 4; // number repetitions

  int rand_seed = 1337;
  int rand_offset = 0;

  T xmax=100.0;         //in mm
  T ymax=100.0;         //  ""
  T zmax=100.0;         //  ""
  T zlevel = 50.0;       // eg zmax/2.0
  T NV=0.0333;          // in 1/mm³
  T mu=0.7;           //in rad^(-1)
  T fdiam=0.2;       // in mm
  T sigmafu=2500.0;      // in MPA
  T d_end=0.3;                //in mm (changed from 3 to 0.3)
  uint n_d=200; //Anzahl der Schrittzahl bei der Spaltöffnung

//  T beta_c = 2.0;
  T weib_m = 10.0;
  T weib_xc = 10.0;

  T fis_k = 20.0;
  T fis_b = 0;// = 2.0/(2.0*k+sqrt(4.0*_params.k*_params.k+4.0));
  T fis_x0 = 0;//= (1.0-b)/(1.0+b);
  T fis_c = 0;// = _params.k*x0+2.0*log(1.0-x0*x0);

};


#endif /* CONFIGURATION_H */
