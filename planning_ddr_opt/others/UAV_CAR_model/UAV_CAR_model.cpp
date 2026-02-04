#include <memory>
#include <math.h>
#include <acado_optimal_control.hpp>
#include <acado_code_generation.hpp>
#include <acado_gnuplot.hpp>
#include <iostream>

// Standalone code generation for a parameter-free quadrotor model
// with thrust and rates input. 

int main( ){
  // Use Acado
  USING_NAMESPACE_ACADO

  /*
  Switch between code generation and analysis.

  If CODE_GEN is true the system is compiled into an optimizaiton problem
  for real-time iteration and all code to run it online is generated.
  Constraints and reference structure is used but the values will be set on
  runtinme.

  If CODE_GEN is false, the system is compiled into a standalone optimization
  and solved on execution. The reference and constraints must be set in here.
  */
  const bool CODE_GEN = true;

  // System variables
  DifferentialState     x, y, psi;
  Control               vr, vl;
  DifferentialEquation  f;
  Function              h, hN;
  OnlineData            xv, yr, yl;
  // double xv = 0.1;
  // double yr = -0.2;
  // double yl = 0.2;

  f << dot(x) == (vr*yl-vl*yr)/(yl-yr)*cos(psi) + (vr-vl)*xv/(yl-yr)*sin(psi);
  f << dot(y) == (vr*yl-vl*yr)/(yl-yr)*sin(psi) - (vr-vl)*xv/(yl-yr)*cos(psi);
  f << dot(psi) == (vr-vl)/(yl-yr);

  // Cost: Sum(i=0, ..., N-1){h_i' * Q * h_i} + h_N' * Q_N * h_N
  h << x << y << psi
    << vr << vl;
    // << dot(vr) << dot(vl);
  hN << x << y << psi;

  DMatrix Q(h.getDim(), h.getDim());
  Q.setIdentity();
  DMatrix QN(hN.getDim(), hN.getDim());
  QN.setIdentity();

  const double t_start = 0.0;
  const double dt = 0.01;
  const double t_end = 0.5;
  const int N = round(t_end/dt);
  OCP ocp(t_start, t_end, N);
  
  
  if(!CODE_GEN)
  {
    Q(0, 0) = 100;
    Q(1, 1) = 100;
    Q(2, 2) = 100;
    QN(0, 0) = 100;
    QN(1, 1) = 100;
    QN(2, 2) = 100;

    // Set a reference for the analysis (if CODE_GEN is false).
    // Reference is at x = 2.0m in hover (qw = 1).
    DVector r(h.getDim());    // Running cost reference
    r.setZero();
    r(0) = 0.0;
    r(1) = 0.0;
    r(2) = M_PI;

    DVector rN(hN.getDim());   // End cost reference
    rN.setZero();
    rN(0) = r(0);
    rN(1) = r(1);
    rN(2) = r(2);

    // For analysis, set references.
    ocp.minimizeLSQ( Q, h, r );
    ocp.minimizeLSQEndTerm( QN, hN, rN );
  }else{
    // For code generation, references are set during run time.
    BMatrix Q_sparse(h.getDim(), h.getDim());
    Q_sparse.setIdentity();
    BMatrix QN_sparse(hN.getDim(), hN.getDim());
    QN_sparse.setIdentity();
    ocp.minimizeLSQ( Q_sparse, h);
    ocp.minimizeLSQEndTerm( QN_sparse, hN );
  }


  const double v_wheel_min = -3.0;
  const double v_wheel_max = 3.0;
  ocp.subjectTo(f);
  ocp.subjectTo(v_wheel_min <= vr <= v_wheel_max);
  ocp.subjectTo(v_wheel_min <= vl <= v_wheel_max);

  ocp.setNOD(3);
  // ocp.setNOD(0);

  if(!CODE_GEN)
  {
    // Set initial state
    ocp.subjectTo( AT_START, x ==  0.0 );
    ocp.subjectTo( AT_START, y ==  0.0 );
    ocp.subjectTo( AT_START, psi ==  0.0 );
    ocp.subjectTo( AT_START, vl ==  0.0 );
    ocp.subjectTo( AT_START, vr ==  0.0 );
    // ocp.subjectTo( AT_START, xv ==  0.1 );
    // ocp.subjectTo( AT_START, yr ==  -0.2 );
    // ocp.subjectTo( AT_START, yl ==  0.2 );

    // Setup some visualization
    GnuplotWindow window1( PLOT_AT_EACH_ITERATION );
    window1.addSubplot( x,"position x" );
    window1.addSubplot( y,"position y" );
    window1.addSubplot( psi,"position psi" );
    window1.addSubplot( vl,"vl" );
    window1.addSubplot( vr,"vr" );


    // Define an algorithm to solve it.
    OptimizationAlgorithm algorithm(ocp);
    algorithm.set( INTEGRATOR_TOLERANCE, 1e-6 );
    algorithm.set( KKT_TOLERANCE, 1e-3 );
    algorithm << window1;
    algorithm.solve();
  }
  else{
    // For code generation, we can set some properties.
    // The main reason for a setting is given as comment.
    OCPexport mpc(ocp);

    mpc.set(HESSIAN_APPROXIMATION,  GAUSS_NEWTON);        // is robust, stable
    mpc.set(DISCRETIZATION_TYPE,    MULTIPLE_SHOOTING);   // good convergence
    mpc.set(SPARSE_QP_SOLUTION,     FULL_CONDENSING_N2);  // due to qpOASES
    mpc.set(INTEGRATOR_TYPE,        INT_IRK_GL4);         // accurate
    mpc.set(NUM_INTEGRATOR_STEPS,   N);
    mpc.set(QP_SOLVER,              QP_QPOASES);          // free, source code
    mpc.set(HOTSTART_QP,            YES);
    mpc.set(CG_USE_OPENMP,                    YES);       // paralellization
    mpc.set(CG_HARDCODE_CONSTRAINT_VALUES,    NO);        // set on runtime
    mpc.set(CG_USE_VARIABLE_WEIGHTING_MATRIX, YES);       // time-varying costs
    mpc.set( USE_SINGLE_PRECISION,        YES);           // Single precision

    // Do not generate tests, makes or matlab-related interfaces.
    mpc.set( GENERATE_TEST_FILE,          NO);
    mpc.set( GENERATE_MAKE_FILE,          NO);
    mpc.set( GENERATE_MATLAB_INTERFACE,   NO);
    mpc.set( GENERATE_SIMULINK_INTERFACE, NO);

    // Finally, export everything.
    if(mpc.exportCode("quadrotor_mpc_codegen") != SUCCESSFUL_RETURN)
      exit( EXIT_FAILURE );
    mpc.printDimensionsQP( );

  }

  return EXIT_SUCCESS;
}