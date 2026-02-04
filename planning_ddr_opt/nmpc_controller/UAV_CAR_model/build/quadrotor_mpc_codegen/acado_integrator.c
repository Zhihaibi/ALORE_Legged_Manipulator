/*
 *    This file was auto-generated using the ACADO Toolkit.
 *    
 *    While ACADO Toolkit is free software released under the terms of
 *    the GNU Lesser General Public License (LGPL), the generated code
 *    as such remains the property of the user who used ACADO Toolkit
 *    to generate this code. In particular, user dependent data of the code
 *    do not inherit the GNU LGPL license. On the other hand, parts of the
 *    generated code that are a direct copy of source code from the
 *    ACADO Toolkit or the software tools it is based on, remain, as derived
 *    work, automatically covered by the LGPL license.
 *    
 *    ACADO Toolkit is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *    
 */


#include "acado_common.h"


real_t rk_dim6_swap;

/** Column vector of size: 6 */
real_t rk_dim6_bPerm[ 6 ];

/** Column vector of size: 13 */
real_t auxVar[ 13 ];

real_t rk_ttt;

/** Row vector of size: 8 */
real_t rk_xxx[ 8 ];

/** Matrix of size: 3 x 2 (row major format) */
real_t rk_kkk[ 6 ];

/** Matrix of size: 6 x 6 (row major format) */
real_t rk_A[ 36 ];

/** Column vector of size: 6 */
real_t rk_b[ 6 ];

/** Row vector of size: 6 */
int rk_dim6_perm[ 6 ];

/** Column vector of size: 3 */
real_t rk_rhsTemp[ 3 ];

/** Matrix of size: 2 x 15 (row major format) */
real_t rk_diffsTemp2[ 30 ];

/** Matrix of size: 3 x 2 (row major format) */
real_t rk_diffK[ 6 ];

/** Matrix of size: 3 x 5 (row major format) */
real_t rk_diffsNew2[ 15 ];

#pragma omp threadprivate( auxVar, rk_ttt, rk_xxx, rk_kkk, rk_diffK, rk_rhsTemp, rk_dim6_perm, rk_A, rk_b, rk_diffsNew2, rk_diffsTemp2, rk_dim6_swap, rk_dim6_bPerm )

void acado_rhs(const real_t* in, real_t* out)
{
const real_t* xd = in;
const real_t* u = in + 3;
const real_t* od = in + 5;
/* Vector of auxiliary variables; number of elements: 4. */
real_t* a = auxVar;

/* Compute intermediate quantities: */
a[0] = (cos(xd[2]));
a[1] = (sin(xd[2]));
a[2] = (sin(xd[2]));
a[3] = (cos(xd[2]));

/* Compute outputs: */
out[0] = (((((u[0]*od[2])-(u[1]*od[1]))/(od[2]-od[1]))*a[0])+((((u[0]-u[1])*od[0])/(od[2]-od[1]))*a[1]));
out[1] = (((((u[0]*od[2])-(u[1]*od[1]))/(od[2]-od[1]))*a[2])-((((u[0]-u[1])*od[0])/(od[2]-od[1]))*a[3]));
out[2] = ((u[0]-u[1])/(od[2]-od[1]));
}



void acado_diffs(const real_t* in, real_t* out)
{
const real_t* xd = in;
const real_t* u = in + 3;
const real_t* od = in + 5;
/* Vector of auxiliary variables; number of elements: 13. */
real_t* a = auxVar;

/* Compute intermediate quantities: */
a[0] = ((real_t)(-1.0000000000000000e+00)*(sin(xd[2])));
a[1] = (cos(xd[2]));
a[2] = ((real_t)(1.0000000000000000e+00)/(od[2]-od[1]));
a[3] = (cos(xd[2]));
a[4] = ((real_t)(1.0000000000000000e+00)/(od[2]-od[1]));
a[5] = (sin(xd[2]));
a[6] = (cos(xd[2]));
a[7] = ((real_t)(-1.0000000000000000e+00)*(sin(xd[2])));
a[8] = ((real_t)(1.0000000000000000e+00)/(od[2]-od[1]));
a[9] = (sin(xd[2]));
a[10] = ((real_t)(1.0000000000000000e+00)/(od[2]-od[1]));
a[11] = (cos(xd[2]));
a[12] = ((real_t)(1.0000000000000000e+00)/(od[2]-od[1]));

/* Compute outputs: */
out[0] = (real_t)(0.0000000000000000e+00);
out[1] = (real_t)(0.0000000000000000e+00);
out[2] = (((((u[0]*od[2])-(u[1]*od[1]))/(od[2]-od[1]))*a[0])+((((u[0]-u[1])*od[0])/(od[2]-od[1]))*a[1]));
out[3] = (((od[2]*a[2])*a[3])+((od[0]*a[4])*a[5]));
out[4] = (((((real_t)(0.0000000000000000e+00)-od[1])*a[2])*a[3])+(((((real_t)(0.0000000000000000e+00)-(real_t)(1.0000000000000000e+00))*od[0])*a[4])*a[5]));
out[5] = (real_t)(0.0000000000000000e+00);
out[6] = (real_t)(0.0000000000000000e+00);
out[7] = (((((u[0]*od[2])-(u[1]*od[1]))/(od[2]-od[1]))*a[6])-((((u[0]-u[1])*od[0])/(od[2]-od[1]))*a[7]));
out[8] = (((od[2]*a[8])*a[9])-((od[0]*a[10])*a[11]));
out[9] = (((((real_t)(0.0000000000000000e+00)-od[1])*a[8])*a[9])-(((((real_t)(0.0000000000000000e+00)-(real_t)(1.0000000000000000e+00))*od[0])*a[10])*a[11]));
out[10] = (real_t)(0.0000000000000000e+00);
out[11] = (real_t)(0.0000000000000000e+00);
out[12] = (real_t)(0.0000000000000000e+00);
out[13] = a[12];
out[14] = (((real_t)(0.0000000000000000e+00)-(real_t)(1.0000000000000000e+00))*a[12]);
}



void acado_solve_dim6_triangular( real_t* const A, real_t* const b )
{

b[5] = b[5]/A[35];
b[4] -= + A[29]*b[5];
b[4] = b[4]/A[28];
b[3] -= + A[23]*b[5];
b[3] -= + A[22]*b[4];
b[3] = b[3]/A[21];
b[2] -= + A[17]*b[5];
b[2] -= + A[16]*b[4];
b[2] -= + A[15]*b[3];
b[2] = b[2]/A[14];
b[1] -= + A[11]*b[5];
b[1] -= + A[10]*b[4];
b[1] -= + A[9]*b[3];
b[1] -= + A[8]*b[2];
b[1] = b[1]/A[7];
b[0] -= + A[5]*b[5];
b[0] -= + A[4]*b[4];
b[0] -= + A[3]*b[3];
b[0] -= + A[2]*b[2];
b[0] -= + A[1]*b[1];
b[0] = b[0]/A[0];
}

real_t acado_solve_dim6_system( real_t* const A, real_t* const b, int* const rk_perm )
{
real_t det;

int i;
int j;
int k;

int indexMax;

int intSwap;

real_t valueMax;

real_t temp;

for (i = 0; i < 6; ++i)
{
rk_perm[i] = i;
}
det = 1.0000000000000000e+00;
for( i=0; i < (5); i++ ) {
	indexMax = i;
	valueMax = fabs(A[i*6+i]);
	for( j=(i+1); j < 6; j++ ) {
		temp = fabs(A[j*6+i]);
		if( temp > valueMax ) {
			indexMax = j;
			valueMax = temp;
		}
	}
	if( indexMax > i ) {
for (k = 0; k < 6; ++k)
{
	rk_dim6_swap = A[i*6+k];
	A[i*6+k] = A[indexMax*6+k];
	A[indexMax*6+k] = rk_dim6_swap;
}
	rk_dim6_swap = b[i];
	b[i] = b[indexMax];
	b[indexMax] = rk_dim6_swap;
	intSwap = rk_perm[i];
	rk_perm[i] = rk_perm[indexMax];
	rk_perm[indexMax] = intSwap;
	}
	det *= A[i*6+i];
	for( j=i+1; j < 6; j++ ) {
		A[j*6+i] = -A[j*6+i]/A[i*6+i];
		for( k=i+1; k < 6; k++ ) {
			A[j*6+k] += A[j*6+i] * A[i*6+k];
		}
		b[j] += A[j*6+i] * b[i];
	}
}
det *= A[35];
det = fabs(det);
acado_solve_dim6_triangular( A, b );
return det;
}

void acado_solve_dim6_system_reuse( real_t* const A, real_t* const b, int* const rk_perm )
{

rk_dim6_bPerm[0] = b[rk_perm[0]];
rk_dim6_bPerm[1] = b[rk_perm[1]];
rk_dim6_bPerm[2] = b[rk_perm[2]];
rk_dim6_bPerm[3] = b[rk_perm[3]];
rk_dim6_bPerm[4] = b[rk_perm[4]];
rk_dim6_bPerm[5] = b[rk_perm[5]];
rk_dim6_bPerm[1] += A[6]*rk_dim6_bPerm[0];

rk_dim6_bPerm[2] += A[12]*rk_dim6_bPerm[0];
rk_dim6_bPerm[2] += A[13]*rk_dim6_bPerm[1];

rk_dim6_bPerm[3] += A[18]*rk_dim6_bPerm[0];
rk_dim6_bPerm[3] += A[19]*rk_dim6_bPerm[1];
rk_dim6_bPerm[3] += A[20]*rk_dim6_bPerm[2];

rk_dim6_bPerm[4] += A[24]*rk_dim6_bPerm[0];
rk_dim6_bPerm[4] += A[25]*rk_dim6_bPerm[1];
rk_dim6_bPerm[4] += A[26]*rk_dim6_bPerm[2];
rk_dim6_bPerm[4] += A[27]*rk_dim6_bPerm[3];

rk_dim6_bPerm[5] += A[30]*rk_dim6_bPerm[0];
rk_dim6_bPerm[5] += A[31]*rk_dim6_bPerm[1];
rk_dim6_bPerm[5] += A[32]*rk_dim6_bPerm[2];
rk_dim6_bPerm[5] += A[33]*rk_dim6_bPerm[3];
rk_dim6_bPerm[5] += A[34]*rk_dim6_bPerm[4];


acado_solve_dim6_triangular( A, rk_dim6_bPerm );
b[0] = rk_dim6_bPerm[0];
b[1] = rk_dim6_bPerm[1];
b[2] = rk_dim6_bPerm[2];
b[3] = rk_dim6_bPerm[3];
b[4] = rk_dim6_bPerm[4];
b[5] = rk_dim6_bPerm[5];
}



/** Matrix of size: 2 x 2 (row major format) */
static const real_t acado_Ah_mat[ 4 ] = 
{ 2.5000000000000001e-03, 5.3867513459481290e-03, 
-3.8675134594812866e-04, 2.5000000000000001e-03 };


/* Fixed step size:0.01 */
int acado_integrate( real_t* const rk_eta, int resetIntegrator )
{
int error;

int i;
int j;
int k;
int run;
int run1;
int tmp_index1;
int tmp_index2;

real_t det;

rk_ttt = 0.0000000000000000e+00;
rk_xxx[3] = rk_eta[18];
rk_xxx[4] = rk_eta[19];
rk_xxx[5] = rk_eta[20];
rk_xxx[6] = rk_eta[21];
rk_xxx[7] = rk_eta[22];

for (run = 0; run < 1; ++run)
{
if( resetIntegrator ) {
for (i = 0; i < 1; ++i)
{
for (run1 = 0; run1 < 2; ++run1)
{
for (j = 0; j < 3; ++j)
{
rk_xxx[j] = rk_eta[j];
tmp_index1 = j;
rk_xxx[j] += + acado_Ah_mat[run1 * 2]*rk_kkk[tmp_index1 * 2];
rk_xxx[j] += + acado_Ah_mat[run1 * 2 + 1]*rk_kkk[tmp_index1 * 2 + 1];
}
acado_diffs( rk_xxx, &(rk_diffsTemp2[ run1 * 15 ]) );
for (j = 0; j < 3; ++j)
{
tmp_index1 = (run1 * 3) + (j);
rk_A[tmp_index1 * 6] = + acado_Ah_mat[run1 * 2]*rk_diffsTemp2[(run1 * 15) + (j * 5)];
rk_A[tmp_index1 * 6 + 1] = + acado_Ah_mat[run1 * 2]*rk_diffsTemp2[(run1 * 15) + (j * 5 + 1)];
rk_A[tmp_index1 * 6 + 2] = + acado_Ah_mat[run1 * 2]*rk_diffsTemp2[(run1 * 15) + (j * 5 + 2)];
if( 0 == run1 ) rk_A[(tmp_index1 * 6) + (j)] -= 1.0000000000000000e+00;
rk_A[tmp_index1 * 6 + 3] = + acado_Ah_mat[run1 * 2 + 1]*rk_diffsTemp2[(run1 * 15) + (j * 5)];
rk_A[tmp_index1 * 6 + 4] = + acado_Ah_mat[run1 * 2 + 1]*rk_diffsTemp2[(run1 * 15) + (j * 5 + 1)];
rk_A[tmp_index1 * 6 + 5] = + acado_Ah_mat[run1 * 2 + 1]*rk_diffsTemp2[(run1 * 15) + (j * 5 + 2)];
if( 1 == run1 ) rk_A[(tmp_index1 * 6) + (j + 3)] -= 1.0000000000000000e+00;
}
acado_rhs( rk_xxx, rk_rhsTemp );
rk_b[run1 * 3] = rk_kkk[run1] - rk_rhsTemp[0];
rk_b[run1 * 3 + 1] = rk_kkk[run1 + 2] - rk_rhsTemp[1];
rk_b[run1 * 3 + 2] = rk_kkk[run1 + 4] - rk_rhsTemp[2];
}
det = acado_solve_dim6_system( rk_A, rk_b, rk_dim6_perm );
for (j = 0; j < 2; ++j)
{
rk_kkk[j] += rk_b[j * 3];
rk_kkk[j + 2] += rk_b[j * 3 + 1];
rk_kkk[j + 4] += rk_b[j * 3 + 2];
}
}
}
for (i = 0; i < 5; ++i)
{
for (run1 = 0; run1 < 2; ++run1)
{
for (j = 0; j < 3; ++j)
{
rk_xxx[j] = rk_eta[j];
tmp_index1 = j;
rk_xxx[j] += + acado_Ah_mat[run1 * 2]*rk_kkk[tmp_index1 * 2];
rk_xxx[j] += + acado_Ah_mat[run1 * 2 + 1]*rk_kkk[tmp_index1 * 2 + 1];
}
acado_rhs( rk_xxx, rk_rhsTemp );
rk_b[run1 * 3] = rk_kkk[run1] - rk_rhsTemp[0];
rk_b[run1 * 3 + 1] = rk_kkk[run1 + 2] - rk_rhsTemp[1];
rk_b[run1 * 3 + 2] = rk_kkk[run1 + 4] - rk_rhsTemp[2];
}
acado_solve_dim6_system_reuse( rk_A, rk_b, rk_dim6_perm );
for (j = 0; j < 2; ++j)
{
rk_kkk[j] += rk_b[j * 3];
rk_kkk[j + 2] += rk_b[j * 3 + 1];
rk_kkk[j + 4] += rk_b[j * 3 + 2];
}
}
for (run1 = 0; run1 < 2; ++run1)
{
for (j = 0; j < 3; ++j)
{
rk_xxx[j] = rk_eta[j];
tmp_index1 = j;
rk_xxx[j] += + acado_Ah_mat[run1 * 2]*rk_kkk[tmp_index1 * 2];
rk_xxx[j] += + acado_Ah_mat[run1 * 2 + 1]*rk_kkk[tmp_index1 * 2 + 1];
}
acado_diffs( rk_xxx, &(rk_diffsTemp2[ run1 * 15 ]) );
for (j = 0; j < 3; ++j)
{
tmp_index1 = (run1 * 3) + (j);
rk_A[tmp_index1 * 6] = + acado_Ah_mat[run1 * 2]*rk_diffsTemp2[(run1 * 15) + (j * 5)];
rk_A[tmp_index1 * 6 + 1] = + acado_Ah_mat[run1 * 2]*rk_diffsTemp2[(run1 * 15) + (j * 5 + 1)];
rk_A[tmp_index1 * 6 + 2] = + acado_Ah_mat[run1 * 2]*rk_diffsTemp2[(run1 * 15) + (j * 5 + 2)];
if( 0 == run1 ) rk_A[(tmp_index1 * 6) + (j)] -= 1.0000000000000000e+00;
rk_A[tmp_index1 * 6 + 3] = + acado_Ah_mat[run1 * 2 + 1]*rk_diffsTemp2[(run1 * 15) + (j * 5)];
rk_A[tmp_index1 * 6 + 4] = + acado_Ah_mat[run1 * 2 + 1]*rk_diffsTemp2[(run1 * 15) + (j * 5 + 1)];
rk_A[tmp_index1 * 6 + 5] = + acado_Ah_mat[run1 * 2 + 1]*rk_diffsTemp2[(run1 * 15) + (j * 5 + 2)];
if( 1 == run1 ) rk_A[(tmp_index1 * 6) + (j + 3)] -= 1.0000000000000000e+00;
}
}
for (run1 = 0; run1 < 3; ++run1)
{
for (i = 0; i < 2; ++i)
{
rk_b[i * 3] = - rk_diffsTemp2[(i * 15) + (run1)];
rk_b[i * 3 + 1] = - rk_diffsTemp2[(i * 15) + (run1 + 5)];
rk_b[i * 3 + 2] = - rk_diffsTemp2[(i * 15) + (run1 + 10)];
}
if( 0 == run1 ) {
det = acado_solve_dim6_system( rk_A, rk_b, rk_dim6_perm );
}
 else {
acado_solve_dim6_system_reuse( rk_A, rk_b, rk_dim6_perm );
}
for (i = 0; i < 2; ++i)
{
rk_diffK[i] = rk_b[i * 3];
rk_diffK[i + 2] = rk_b[i * 3 + 1];
rk_diffK[i + 4] = rk_b[i * 3 + 2];
}
for (i = 0; i < 3; ++i)
{
rk_diffsNew2[(i * 5) + (run1)] = (i == run1-0);
rk_diffsNew2[(i * 5) + (run1)] += + rk_diffK[i * 2]*(real_t)5.0000000000000001e-03 + rk_diffK[i * 2 + 1]*(real_t)5.0000000000000001e-03;
}
}
for (run1 = 0; run1 < 2; ++run1)
{
for (i = 0; i < 2; ++i)
{
for (j = 0; j < 3; ++j)
{
tmp_index1 = (i * 3) + (j);
tmp_index2 = (run1) + (j * 5);
rk_b[tmp_index1] = - rk_diffsTemp2[(i * 15) + (tmp_index2 + 3)];
}
}
acado_solve_dim6_system_reuse( rk_A, rk_b, rk_dim6_perm );
for (i = 0; i < 2; ++i)
{
rk_diffK[i] = rk_b[i * 3];
rk_diffK[i + 2] = rk_b[i * 3 + 1];
rk_diffK[i + 4] = rk_b[i * 3 + 2];
}
for (i = 0; i < 3; ++i)
{
rk_diffsNew2[(i * 5) + (run1 + 3)] = + rk_diffK[i * 2]*(real_t)5.0000000000000001e-03 + rk_diffK[i * 2 + 1]*(real_t)5.0000000000000001e-03;
}
}
rk_eta[0] += + rk_kkk[0]*(real_t)5.0000000000000001e-03 + rk_kkk[1]*(real_t)5.0000000000000001e-03;
rk_eta[1] += + rk_kkk[2]*(real_t)5.0000000000000001e-03 + rk_kkk[3]*(real_t)5.0000000000000001e-03;
rk_eta[2] += + rk_kkk[4]*(real_t)5.0000000000000001e-03 + rk_kkk[5]*(real_t)5.0000000000000001e-03;
for (i = 0; i < 3; ++i)
{
for (j = 0; j < 3; ++j)
{
tmp_index2 = (j) + (i * 3);
rk_eta[tmp_index2 + 3] = rk_diffsNew2[(i * 5) + (j)];
}
for (j = 0; j < 2; ++j)
{
tmp_index2 = (j) + (i * 2);
rk_eta[tmp_index2 + 12] = rk_diffsNew2[(i * 5) + (j + 3)];
}
}
resetIntegrator = 0;
rk_ttt += 1.0000000000000000e+00;
}
for (i = 0; i < 3; ++i)
{
}
if( det < 1e-12 ) {
error = 2;
} else if( det < 1e-6 ) {
error = 1;
} else {
error = 0;
}
return error;
}



