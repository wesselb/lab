*
* This file contains a test program and functions TVTL (trivariate normal
* and t), BVTL (bivariate t), BVND (bivariate normal), STUDNT (univariate
* t), PHID (univariate normal), plus some support functions.
* The file is self contained and should compile without errors on (77)
* standard Fortran compilers. The test program demonstrates the use of
* TVTL for computing trivariate distribution values 20 test problems
* with NU = 0 (normal case), 3, 6, 9, and 12.
*
* The software is based on work described in the paper
*  "Numerical Computation of Rectangular Bivariate and Trivariate Normal
*    and t Probabilities", by
*
*          Alan Genz
*          Department of Mathematics
*          Washington State University
*          Pullman, WA 99164-3113
*          Email : alangenz@wsu.edu
*
      PROGRAM TVTST
      INTEGER I, J, NU, NT
      PARAMETER ( NT = 20 )
      DOUBLE PRECISION TVTL, LIMIT(3,NT), SIGMA(3,NT), EPS, V
      DATA ( LIMIT(I,1), I = 1, 3 ), ( SIGMA(I,1), I = 1, 3 )
     &     / .5D0, .5D0, .8D0, .1D0, .6D0, .8D0 /
      DATA ( LIMIT(I,2), I = 1, 3 ), ( SIGMA(I,2), I = 1, 3 )
     &     / -2.5D0, .5D0, .8D0, .1D0, -.6D0, -.8D0 /
      DATA ( LIMIT(I,3), I = 1, 3 ), ( SIGMA(I,3), I = 1, 3 )
     &     / 1.5D0, .5D0, .8D0, .1D0, .6D0, .8D0 /
      DATA ( LIMIT(I,4), I = 1, 3 ), ( SIGMA(I,4), I = 1, 3 )
     &     / .5D0, .5D0, .8D0, .1D0, -.6D0, -.8D0 /
      DATA ( LIMIT(I,5), I = 1, 3 ), ( SIGMA(I,5), I = 1, 3 )
     &     / .5D0, .5D0, .8D0, .1D0, -.5D0, .5D0 /
      DATA ( LIMIT(I,6), I = 1, 3 ), ( SIGMA(I,6), I = 1, 3 )
     &     / -1.5D0, .5D0, .8D0, .1D0, -.5D0, .5D0 /
      DATA ( LIMIT(I,7), I = 1, 3 ), ( SIGMA(I,7), I = 1, 3 )
     &     / 1.5D0, .5D0, .8D0, .1D0, .5D0, -.5D0 /
      DATA ( LIMIT(I,8), I = 1, 3 ), ( SIGMA(I,8), I = 1, 3 )
     &     / -.5D0, 1D0, 1.2D0, -.4D0, .2D0, .7D0 /
      DATA ( LIMIT(I,9), I = 1, 3 ), ( SIGMA(I,9), I = 1, 3 )
     &     / 1D0, 1D0, 2D0, .4D0, .8D0, .8D0 /
      DATA ( LIMIT(I,10), I = 1, 3 ), ( SIGMA(I,10), I = 1, 3 )
     &     / 1D0, 2D0, 1D0, .4D0, .8D0, .8D0 /
      DATA ( LIMIT(I,11), I = 1, 3 ), ( SIGMA(I,11), I = 1, 3 )
     &     / -2D0, -2D0, -2D0, .4D0, .8D0, .8D0 /
      DATA ( LIMIT(I,12), I = 1, 3 ), ( SIGMA(I,12), I = 1, 3 )
     *     / 1D0, 2D0, 3D0, -.998D0, -0.248D0, 0.248D0 /
      DATA ( LIMIT(I,13), I = 1, 3 ), ( SIGMA(I,13), I = 1, 3 )
     *     / -1D0, 2D0, 3D0, .25D0, 0.25D0, 0.25D0 /
      DATA ( LIMIT(I,14), I = 1, 3 ), ( SIGMA(I,14), I = 1, 3 )
     *     /  1D0, 1D0, 3D0, .998D0, 0.2482D0, 0.2487D0 /
      DATA ( LIMIT(I,15), I = 1, 3 ), ( SIGMA(I,15), I = 1, 3 )
     *     /  1D0, 1D0, 3D0, .998D0, 0.5D0, 0.5D0 /
      DATA ( LIMIT(I,16), I = 1, 3 ), ( SIGMA(I,16), I = 1, 3 )
     *     /  1D0, 1D0, 3D0, .99D0, 0.99D0, 0.99D0 /
      DATA ( LIMIT(I,17), I = 1, 3 ), ( SIGMA(I,17), I = 1, 3 )
     *     /  1D0, 2D0, 3D0, -1D0, -.99D0, .99D0 /
      DATA ( LIMIT(I,18), I = 1, 3 ), ( SIGMA(I,18), I = 1, 3 )
     *     /  1D0, 2D0, 3D0, 1D0, -.99D0, -.99D0 /
      DATA ( LIMIT(I,19), I = 1, 3 ), ( SIGMA(I,19), I = 1, 3 )
     *     /  1D0, -1D0, 1D0, .998D0, -0.2482D0, -0.2482D0 /
      DATA ( LIMIT(I,NT), I = 1, 3 ), ( SIGMA(I,NT), I = 1, 3 )
     *     /  1D0, -1D0, 2D0, .99992D0, 0.64627D0, 0.63975D0 /
      EPS = 1D-6
      PRINT '(''      Trivariate t Test with EPS ='', E10.1)', EPS
      DO NU = 0, 12, 3
         PRINT '(''NU   B1   B2   B3    R21      R31      R32    TVT'')'
         DO J = 1, NT
            V = TVTL( NU, LIMIT(1,J), SIGMA(1,J), EPS )
            PRINT '(I2,3F5.1,3F9.5,F13.10)', NU,
     &           ( LIMIT(I,J), I = 1, 3 ), ( SIGMA(I,J), I = 1, 3 ), V
         END DO
      END DO
      END
*
      DOUBLE PRECISION FUNCTION TVTL( NU, H, R, EPSI )
*
*     A function for computing trivariate normal and t-probabilities.
*     This function uses algorithms developed from the ideas
*     described in the papers:
*       R.L. Plackett, Biometrika 41(1954), pp. 351-360.
*       Z. Drezner, Math. Comp. 62(1994), pp. 289-294.
*     with adaptive integration from (0,0,1) to (0,0,r23) to R.
*
*      Calculate the probability that X(I) < H(I), for I = 1,2,3
*    NU   INTEGER degrees of freedom; use NU = 0 for normal cases.
*    H    REAL array of uppoer limits for probability distribution
*    R    REAL array of three correlation coefficients, R should
*         contain the lower left portion of the correlation matrix r.
*         R should contains the values r21, r31, r23 in that order.
*   EPSI  REAL required absolute accuracy; maximum accuracy for most
*          computations is approximately 1D-14
*
*    The software is based on work described in the paper
*     "Numerical Computation of Rectangular Bivariate and Trivariate
*      Normal and t Probabilities", by the code author:
*
*       Alan Genz
*       Department of Mathematics
*       Washington State University
*       Pullman, WA 99164-3113
*       Email : alangenz@wsu.edu
*
*
* Copyright (C) 2013, Alan Genz,  All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided the following conditions are met:
*   1. Redistributions of source code must retain the above copyright
*      notice, this list of conditions and the following disclaimer.
*   2. Redistributions in binary form must reproduce the above copyright
*      notice, this list of conditions and the following disclaimer in
*      the documentation and/or other materials provided with the
*      distribution.
*   3. The contributor name(s) may not be used to endorse or promote
*      products derived from this software without specific prior
*      written permission.
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
* COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
* OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
* TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
      EXTERNAL TVTMFN
      INTEGER NU, NUC
      DOUBLE PRECISION H(3), H1, H2, H3, R(3), R12, R13, R23, EPSI
      DOUBLE PRECISION ONE, ZRO, EPS, ZROS(3), HS(3), TVT
      DOUBLE PRECISION RUA, RUB, AR, RUC, PT, BVTL, PHID, ADONET
      PARAMETER ( ZRO = 0, ONE = 1 )
      COMMON /TVTMBK/ H1, H2, H3, R23, RUA, RUB, AR, RUC, NUC
      EPS = MAX( 1D-14, EPSI )
      PT = ASIN(ONE)
      NUC = NU
      H1 = H(1)
      H2 = H(2)
      H3 = H(3)
      R12 = R(1)
      R13 = R(2)
      R23 = R(3)
*
*     Sort R's and check for special cases
*
      IF ( ABS(R12) .GT. ABS(R13) ) THEN
         H2 = H3
         H3 = H(2)
         R12 = R13
         R13 = R(1)
      END IF
      IF ( ABS(R13) .GT. ABS(R23) ) THEN
         H1 = H2
         H2 = H(1)
         R23 = R13
         R13 = R(3)
      END IF
      TVT = 0
      IF ( ABS(H1) + ABS(H2) + ABS(H3) .LT. EPS ) THEN
         TVT = ( 1 + ( ASIN(R12) + ASIN(R13) + ASIN(R23) )/PT )/8
      ELSE IF ( NU .LT. 1 .AND. ABS(R12) + ABS(R13) .LT. EPS ) THEN
         TVT = PHID(H1)*BVTL( NU, H2, H3, R23 )
      ELSE IF ( NU .LT. 1 .AND. ABS(R13) + ABS(R23) .LT. EPS ) THEN
         TVT = PHID(H3)*BVTL( NU, H1, H2, R12 )
      ELSE IF ( NU .LT. 1 .AND. ABS(R12) + ABS(R23) .LT. EPS ) THEN
         TVT = PHID(H2)*BVTL( NU, H1, H3, R13 )
      ELSE IF ( 1 - R23 .LT. EPS ) THEN
         TVT = BVTL( NU, H1, MIN( H2, H3 ), R12 )
      ELSE IF ( R23 + 1 .LT. EPS ) THEN
         IF  ( H2 .GT. -H3 )
     &        TVT = BVTL( NU, H1, H2, R12 ) - BVTL( NU, H1, -H3, R12 )
      ELSE
*
*        Compute singular TVT value
*
         IF ( NU .LT. 1 ) THEN
            TVT = BVTL( NU, H2, H3, R23 )*PHID(H1)
         ELSE IF ( R23 .GE. 0 ) THEN
            TVT = BVTL( NU, H1, MIN( H2, H3 ), ZRO )
         ELSE IF ( H2 .GT. -H3 ) THEN
            TVT = BVTL( NU, H1, H2, ZRO ) - BVTL( NU, H1, -H3, ZRO )
         END IF
*
*        Use numerical integration to compute probability
*
*
         RUA = ASIN( R12 )
         RUB = ASIN( R13 )
         AR = ASIN( R23)
         RUC = SIGN( PT, AR ) - AR
         TVT = TVT + ADONET( TVTMFN, ZRO, ONE, EPS )/( 4*PT )
      END IF
      TVTL = MAX( ZRO, MIN( TVT, ONE ) )
      END
*
      DOUBLE PRECISION FUNCTION TVTMFN( X )
*
*     Computes Plackett formula integrands
*
      INTEGER NU
      DOUBLE PRECISION X, H1, H2, H3, R23, RUA, RUB, AR, RUC
      DOUBLE PRECISION R12, RR2, R13, RR3, R, RR, ZRO, PNTGND
      PARAMETER ( ZRO = 0 )
      COMMON /TVTMBK/ H1, H2, H3, R23, RUA, RUB, AR, RUC, NU
      TVTMFN = 0
      CALL SINCS( RUA*X, R12, RR2 )
      CALL SINCS( RUB*X, R13, RR3 )
      IF ( ABS(RUA) .GT. 0 )
     &     TVTMFN = TVTMFN + RUA*PNTGND( NU, H1,H2,H3, R13,R23,R12,RR2 )
      IF ( ABS(RUB) .GT. 0 )
     &     TVTMFN = TVTMFN + RUB*PNTGND( NU, H1,H3,H2, R12,R23,R13,RR3 )
      IF ( NU .GT. 0 ) THEN
         CALL SINCS( AR + RUC*X, R, RR )
         TVTMFN = TVTMFN - RUC*PNTGND( NU, H2, H3, H1, ZRO, ZRO, R, RR )
      END IF
      END
*
      SUBROUTINE SINCS( X, SX, CS )
*
*     Computes SIN(X), COS(X)^2, with series approx. for |X| near PI/2
*
      DOUBLE PRECISION X, SX, CS, PT, EE
      PARAMETER ( PT = 1.57079632679489661923132169163975D0 )
      EE = ( PT - ABS(X) )**2
      IF ( EE .LT. 5D-5 ) THEN
         SX = SIGN( 1 - EE*( 1 - EE/12 )/2, X )
         CS = EE*( 1 - EE*( 1 - 2*EE/15 )/3 )
      ELSE
         SX = SIN(X)
         CS = 1 - SX*SX
      END IF
      END
*
      DOUBLE PRECISION FUNCTION PNTGND( NU, BA, BB, BC, RA, RB, R, RR )
*
*     Computes Plackett formula integrand
*
      INTEGER NU
      DOUBLE PRECISION BA, BB, BC, RA, RB, R, RR
      DOUBLE PRECISION DT, FT, BT, PHID, STUDNT
      PNTGND = 0
      DT = RR*( RR - ( RA - RB )**2 - 2*RA*RB*( 1 - R ) )
      IF ( DT .GT. 0 ) THEN
         BT = ( BC*RR + BA*( R*RB - RA ) + BB*( R*RA -RB ) )/SQRT(DT)
         FT = ( BA - R*BB )**2/RR + BB*BB
         IF ( NU .LT. 1 ) THEN
            IF ( BT .GT. -10 .AND. FT .LT. 100 ) THEN
               PNTGND = EXP( -FT/2 )
               IF ( BT .LT. 10 ) PNTGND = PNTGND*PHID(BT)
            END IF
         ELSE
            FT = SQRT( 1 + FT/NU )
            PNTGND = STUDNT( NU, BT/FT )/FT**NU
         END IF
      END IF
      END
*
      DOUBLE PRECISION FUNCTION ADONET( F, A, B, TOL )
*
*     One Dimensional Globally Adaptive Integration Function
*
      EXTERNAL F
      DOUBLE PRECISION F, A, B, TOL
      INTEGER NL, I, IM, IP
      PARAMETER ( NL = 100 )
      DOUBLE PRECISION EI(NL), AI(NL), BI(NL), FI(NL), FIN, ERR, KRNRDT
      COMMON /ABLK/ ERR, IM
      AI(1) = A
      BI(1) = B
      ERR = 1
      IP = 1
      IM = 1
      DO WHILE ( 4*ERR .GT. TOL .AND. IM .LT. NL )
         IM = IM + 1
         BI(IM) = BI(IP)
         AI(IM) = ( AI(IP) + BI(IP) )/2
         BI(IP) = AI(IM)
         FI(IP) = KRNRDT( AI(IP), BI(IP), F, EI(IP) )
         FI(IM) = KRNRDT( AI(IM), BI(IM), F, EI(IM) )
         ERR = 0
         FIN = 0
         DO I = 1, IM
            IF ( EI(I) .GT. EI(IP) ) IP = I
            FIN = FIN + FI(I)
            ERR = ERR + EI(I)**2
         END DO
         ERR = SQRT( ERR )
      END DO
      ADONET = FIN
      END
*
      DOUBLE PRECISION FUNCTION KRNRDT( A, B, F, ERR )
*
*     Kronrod Rule
*
      DOUBLE PRECISION A, B, ERR, T, CEN, F, FC, WID, RESG, RESK
*
*        The abscissae and weights are given for the interval (-1,1);
*        only positive abscissae and corresponding weights are given.
*
*        XGK    - abscissae of the 2N+1-point Kronrod rule:
*                 XGK(2), XGK(4), ...  N-point Gauss rule abscissae;
*                 XGK(1), XGK(3), ...  optimally added abscissae.
*        WGK    - weights of the 2N+1-point Kronrod rule.
*        WG     - weights of the N-point Gauss rule.
*
      INTEGER J, N
      PARAMETER ( N = 11 )
      DOUBLE PRECISION WG(0:(N+1)/2), WGK(0:N), XGK(0:N)
      SAVE WG, WGK, XGK
      DATA WG( 0)/ 0.2729250867779007D+00/
      DATA WG( 1)/ 0.5566856711617449D-01/
      DATA WG( 2)/ 0.1255803694649048D+00/
      DATA WG( 3)/ 0.1862902109277352D+00/
      DATA WG( 4)/ 0.2331937645919914D+00/
      DATA WG( 5)/ 0.2628045445102478D+00/
*
      DATA XGK( 0)/ 0.0000000000000000D+00/
      DATA XGK( 1)/ 0.9963696138895427D+00/
      DATA XGK( 2)/ 0.9782286581460570D+00/
      DATA XGK( 3)/ 0.9416771085780681D+00/
      DATA XGK( 4)/ 0.8870625997680953D+00/
      DATA XGK( 5)/ 0.8160574566562211D+00/
      DATA XGK( 6)/ 0.7301520055740492D+00/
      DATA XGK( 7)/ 0.6305995201619651D+00/
      DATA XGK( 8)/ 0.5190961292068118D+00/
      DATA XGK( 9)/ 0.3979441409523776D+00/
      DATA XGK(10)/ 0.2695431559523450D+00/
      DATA XGK(11)/ 0.1361130007993617D+00/
*
      DATA WGK( 0)/ 0.1365777947111183D+00/
      DATA WGK( 1)/ 0.9765441045961290D-02/
      DATA WGK( 2)/ 0.2715655468210443D-01/
      DATA WGK( 3)/ 0.4582937856442671D-01/
      DATA WGK( 4)/ 0.6309742475037484D-01/
      DATA WGK( 5)/ 0.7866457193222764D-01/
      DATA WGK( 6)/ 0.9295309859690074D-01/
      DATA WGK( 7)/ 0.1058720744813894D+00/
      DATA WGK( 8)/ 0.1167395024610472D+00/
      DATA WGK( 9)/ 0.1251587991003195D+00/
      DATA WGK(10)/ 0.1312806842298057D+00/
      DATA WGK(11)/ 0.1351935727998845D+00/
*
*           Major variables
*
*           CEN  - mid point of the interval
*           WID  - half-length of the interval
*           RESG - result of the N-point Gauss formula
*           RESK - result of the 2N+1-point Kronrod formula
*
*           Compute the 2N+1-point Kronrod approximation to
*            the integral, and estimate the absolute error.
*
      WID = ( B - A )/2
      CEN = ( B + A )/2
      FC = F(CEN)
      RESG = FC*WG(0)
      RESK = FC*WGK(0)
      DO J = 1, N
         T = WID*XGK(J)
         FC = F( CEN - T ) + F( CEN + T )
         RESK = RESK + WGK(J)*FC
         IF( MOD( J, 2 ) .EQ. 0 ) RESG = RESG + WG(J/2)*FC
      END DO
      KRNRDT = WID*RESK
      ERR = ABS( WID*( RESK - RESG ) )
      END
*
      DOUBLE PRECISION FUNCTION STUDNT( NU, T )
*
*     Student t Distribution Function
*
*                       T
*         STUDNT = C   I  ( 1 + y*y/NU )**( -(NU+1)/2 ) dy
*                   NU -INF
*
      INTEGER NU, J
      DOUBLE PRECISION T, ZRO, ONE, PI, PHID
      DOUBLE PRECISION CSSTHE, SNTHE, POLYN, TT, TS, RN
      PARAMETER ( ZRO = 0, ONE = 1 )
      PI = ACOS(-ONE)
      IF ( NU .LT. 1 ) THEN
         STUDNT = PHID( T )
      ELSE IF ( NU .EQ. 1 ) THEN
         STUDNT = ( 1 + 2*ATAN(T)/PI )/2
      ELSE IF ( NU .EQ. 2 ) THEN
         STUDNT = ( 1 + T/SQRT( 2 + T*T ))/2
      ELSE
         TT = T*T
         CSSTHE = 1/( 1 + TT/NU )
         POLYN = 1
         DO J = NU-2, 2, -2
            POLYN = 1 + ( J - 1 )*CSSTHE*POLYN/J
         END DO
         IF ( MOD( NU, 2 ) .EQ. 1 ) THEN
            RN = NU
            TS = T/SQRT(RN)
            STUDNT = ( 1 + 2*( ATAN(TS) + TS*CSSTHE*POLYN )/PI )/2
         ELSE
            SNTHE = T/SQRT( NU + TT )
            STUDNT = ( 1 + SNTHE*POLYN )/2
         END IF
         STUDNT = MAX( ZRO, MIN( STUDNT, ONE ) )
      ENDIF
      END
*
      DOUBLE PRECISION FUNCTION BVTL( NU, DH, DK, R )
*
*     A function for computing bivariate t probabilities.
*
*       Alan Genz
*       Department of Mathematics
*       Washington State University
*       Pullman, WA 99164-3113
*       Email : alangenz@wsu.edu
*
*    This function is based on the method described by
*        Dunnett, C.W. and M. Sobel, (1954),
*        A bivariate generalization of Student's t-distribution
*        with tables for certain special cases,
*        Biometrika 41, pp. 153-169.
*
* BVTL - calculate the probability that X < DH and Y < DK.
*
* parameters
*
*   NU number of degrees of freedom
*   DH 1st lower integration limit
*   DK 2nd lower integration limit
*   R   correlation coefficient
*
      INTEGER NU, J, HS, KS
      DOUBLE PRECISION DH, DK, R
      DOUBLE PRECISION TPI, PI, ORS, HRK, KRH, BVT, SNU, BVND, STUDNT
      DOUBLE PRECISION GMPH, GMPK, XNKH, XNHK, QHRK, HKN, HPK, HKRN
      DOUBLE PRECISION BTNCKH, BTNCHK, BTPDKH, BTPDHK, ONE, EPS
      PARAMETER ( ONE = 1, EPS = 1D-15 )
      IF ( NU .LT. 1 ) THEN
         BVTL = BVND( -DH, -DK, R )
      ELSE IF ( 1 - R .LE. EPS ) THEN
            BVTL = STUDNT( NU, MIN( DH, DK ) )
      ELSE IF ( R + 1  .LE. EPS ) THEN
         IF ( DH .GT. -DK )  THEN
            BVTL = STUDNT( NU, DH ) - STUDNT( NU, -DK )
         ELSE
            BVTL = 0
         END IF
      ELSE
         PI = ACOS(-ONE)
         TPI = 2*PI
         SNU = NU
         SNU = SQRT(SNU)
         ORS = 1 - R*R
         HRK = DH - R*DK
         KRH = DK - R*DH
         IF ( ABS(HRK) + ORS .GT. 0 ) THEN
            XNHK = HRK**2/( HRK**2 + ORS*( NU + DK**2 ) )
            XNKH = KRH**2/( KRH**2 + ORS*( NU + DH**2 ) )
         ELSE
            XNHK = 0
            XNKH = 0
         END IF
         HS = SIGN( ONE, DH - R*DK )
         KS = SIGN( ONE, DK - R*DH )
         IF ( MOD( NU, 2 ) .EQ. 0 ) THEN
            BVT = ATAN2( SQRT(ORS), -R )/TPI
            GMPH = DH/SQRT( 16*( NU + DH**2 ) )
            GMPK = DK/SQRT( 16*( NU + DK**2 ) )
            BTNCKH = 2*ATAN2( SQRT( XNKH ), SQRT( 1 - XNKH ) )/PI
            BTPDKH = 2*SQRT( XNKH*( 1 - XNKH ) )/PI
            BTNCHK = 2*ATAN2( SQRT( XNHK ), SQRT( 1 - XNHK ) )/PI
            BTPDHK = 2*SQRT( XNHK*( 1 - XNHK ) )/PI
            DO J = 1, NU/2
               BVT = BVT + GMPH*( 1 + KS*BTNCKH )
               BVT = BVT + GMPK*( 1 + HS*BTNCHK )
               BTNCKH = BTNCKH + BTPDKH
               BTPDKH = 2*J*BTPDKH*( 1 - XNKH )/( 2*J + 1 )
               BTNCHK = BTNCHK + BTPDHK
               BTPDHK = 2*J*BTPDHK*( 1 - XNHK )/( 2*J + 1 )
               GMPH = GMPH*( 2*J - 1 )/( 2*J*( 1 + DH**2/NU ) )
               GMPK = GMPK*( 2*J - 1 )/( 2*J*( 1 + DK**2/NU ) )
            END DO
         ELSE
            QHRK = SQRT( DH**2 + DK**2 - 2*R*DH*DK + NU*ORS )
            HKRN = DH*DK + R*NU
            HKN = DH*DK - NU
            HPK = DH + DK
            BVT = ATAN2( -SNU*( HKN*QHRK + HPK*HKRN ),
     &                          HKN*HKRN-NU*HPK*QHRK )/TPI
            IF ( BVT .LT. -EPS ) BVT = BVT + 1
            GMPH = DH/( TPI*SNU*( 1 + DH**2/NU ) )
            GMPK = DK/( TPI*SNU*( 1 + DK**2/NU ) )
            BTNCKH = SQRT( XNKH )
            BTPDKH = BTNCKH
            BTNCHK = SQRT( XNHK )
            BTPDHK = BTNCHK
            DO J = 1, ( NU - 1 )/2
               BVT = BVT + GMPH*( 1 + KS*BTNCKH )
               BVT = BVT + GMPK*( 1 + HS*BTNCHK )
               BTPDKH = ( 2*J - 1 )*BTPDKH*( 1 - XNKH )/( 2*J )
               BTNCKH = BTNCKH + BTPDKH
               BTPDHK = ( 2*J - 1 )*BTPDHK*( 1 - XNHK )/( 2*J )
               BTNCHK = BTNCHK + BTPDHK
               GMPH = 2*J*GMPH/( ( 2*J + 1 )*( 1 + DH**2/NU ) )
               GMPK = 2*J*GMPK/( ( 2*J + 1 )*( 1 + DK**2/NU ) )
            END DO
         END IF
         BVTL = BVT
      END IF
*     END BVTL
      END
*
      DOUBLE PRECISION FUNCTION PHID(Z)
*
*     Normal distribution probabilities accurate to 1d-15.
*     Reference: J.L. Schonfelder, Math Comp 32(1978), pp 1232-1240.
*
      INTEGER I, IM
      DOUBLE PRECISION A(0:43), BM, B, BP, P, RTWO, T, XA, Z
      PARAMETER( RTWO = 1.414213562373095048801688724209D0, IM = 24 )
      SAVE A
      DATA ( A(I), I = 0, 43 )/
     &    6.10143081923200417926465815756D-1,
     &   -4.34841272712577471828182820888D-1,
     &    1.76351193643605501125840298123D-1,
     &   -6.0710795609249414860051215825D-2,
     &    1.7712068995694114486147141191D-2,
     &   -4.321119385567293818599864968D-3,
     &    8.54216676887098678819832055D-4,
     &   -1.27155090609162742628893940D-4,
     &    1.1248167243671189468847072D-5, 3.13063885421820972630152D-7,
     &   -2.70988068537762022009086D-7, 3.0737622701407688440959D-8,
     &    2.515620384817622937314D-9, -1.028929921320319127590D-9,
     &    2.9944052119949939363D-11, 2.6051789687266936290D-11,
     &   -2.634839924171969386D-12, -6.43404509890636443D-13,
     &    1.12457401801663447D-13, 1.7281533389986098D-14,
     &   -4.264101694942375D-15, -5.45371977880191D-16,
     &    1.58697607761671D-16, 2.0899837844334D-17,
     &   -5.900526869409D-18, -9.41893387554D-19, 2.14977356470D-19,
     &    4.6660985008D-20, -7.243011862D-21, -2.387966824D-21,
     &    1.91177535D-22, 1.20482568D-22, -6.72377D-25, -5.747997D-24,
     &   -4.28493D-25, 2.44856D-25, 4.3793D-26, -8.151D-27, -3.089D-27,
     &    9.3D-29, 1.74D-28, 1.6D-29, -8.0D-30, -2.0D-30 /
*
      XA = ABS(Z)/RTWO
      IF ( XA .GT. 100 ) THEN
         P = 0
      ELSE
         T = ( 8*XA - 30 ) / ( 4*XA + 15 )
         BM = 0
         B  = 0
         DO I = IM, 0, -1
            BP = B
            B  = BM
            BM = T*B - BP  + A(I)
         END DO
         P = EXP( -XA*XA )*( BM - BP )/4
      END IF
      IF ( Z .GT. 0 ) P = 1 - P
      PHID = P
      END
*
      DOUBLE PRECISION FUNCTION BVND( DH, DK, R )
*
*     A function for computing bivariate normal probabilities.
*
*       Alan Genz
*       Department of Mathematics
*       Washington State University
*       Pullman, WA 99164-3113
*       Email : alangenz@wsu.edu
*
*    This function is based on the method described by
*        Drezner, Z and G.O. Wesolowsky, (1989),
*        On the computation of the bivariate normal integral,
*        Journal of Statist. Comput. Simul. 35, pp. 101-107,
*    with major modifications for double precision, and for |R| close to 1.
*
* BVND calculates the probability that X > DH and Y > DK.
*      Note: Prob( X < DH, Y < DK ) = BVND( -DH, -DK, R ).
*
* Parameters
*
*   DH  DOUBLE PRECISION, integration limit
*   DK  DOUBLE PRECISION, integration limit
*   R   DOUBLE PRECISION, correlation coefficient
*
      DOUBLE PRECISION DH, DK, R, TWOPI
      INTEGER I, IS, LG, NG
      PARAMETER ( TWOPI = 6.283185307179586D0 )
      DOUBLE PRECISION X(10,3), W(10,3), AS, A, B, C, D, RS, XS, BVN
      DOUBLE PRECISION PHID, SN, ASR, H, K, BS, HS, HK
*     Gauss Legendre Points and Weights, N =  6
      DATA ( W(I,1), X(I,1), I = 1,3) /
     &  0.1713244923791705D+00,-0.9324695142031522D+00,
     &  0.3607615730481384D+00,-0.6612093864662647D+00,
     &  0.4679139345726904D+00,-0.2386191860831970D+00/
*     Gauss Legendre Points and Weights, N = 12
      DATA ( W(I,2), X(I,2), I = 1,6) /
     &  0.4717533638651177D-01,-0.9815606342467191D+00,
     &  0.1069393259953183D+00,-0.9041172563704750D+00,
     &  0.1600783285433464D+00,-0.7699026741943050D+00,
     &  0.2031674267230659D+00,-0.5873179542866171D+00,
     &  0.2334925365383547D+00,-0.3678314989981802D+00,
     &  0.2491470458134029D+00,-0.1252334085114692D+00/
*     Gauss Legendre Points and Weights, N = 20
      DATA ( W(I,3), X(I,3), I = 1, 10 ) /
     &  0.1761400713915212D-01,-0.9931285991850949D+00,
     &  0.4060142980038694D-01,-0.9639719272779138D+00,
     &  0.6267204833410906D-01,-0.9122344282513259D+00,
     &  0.8327674157670475D-01,-0.8391169718222188D+00,
     &  0.1019301198172404D+00,-0.7463319064601508D+00,
     &  0.1181945319615184D+00,-0.6360536807265150D+00,
     &  0.1316886384491766D+00,-0.5108670019508271D+00,
     &  0.1420961093183821D+00,-0.3737060887154196D+00,
     &  0.1491729864726037D+00,-0.2277858511416451D+00,
     &  0.1527533871307259D+00,-0.7652652113349733D-01/
      SAVE X, W
      IF ( ABS(R) .LT. 0.3 ) THEN
         NG = 1
         LG = 3
      ELSE IF ( ABS(R) .LT. 0.75 ) THEN
         NG = 2
         LG = 6
      ELSE
         NG = 3
         LG = 10
      ENDIF
      H = DH
      K = DK
      HK = H*K
      BVN = 0
      IF ( ABS(R) .LT. 0.925 ) THEN
         IF ( ABS(R) .GT. 0 ) THEN
            HS = ( H*H + K*K )/2
            ASR = ASIN(R)
            DO I = 1, LG
               DO IS = -1, 1, 2
                  SN = SIN( ASR*(  IS*X(I,NG) + 1 )/2 )
                  BVN = BVN + W(I,NG)*EXP( ( SN*HK-HS )/( 1-SN*SN ) )
               END DO
            END DO
            BVN = BVN*ASR/( 2*TWOPI )
         ENDIF
         BVN = BVN + PHID(-H)*PHID(-K)
      ELSE
         IF ( R .LT. 0 ) THEN
            K = -K
            HK = -HK
         ENDIF
         IF ( ABS(R) .LT. 1 ) THEN
            AS = ( 1 - R )*( 1 + R )
            A = SQRT(AS)
            BS = ( H - K )**2
            C = ( 4 - HK )/8
            D = ( 12 - HK )/16
            ASR = -( BS/AS + HK )/2
            IF ( ASR .GT. -100 ) BVN = A*EXP(ASR)
     &             *( 1 - C*( BS - AS )*( 1 - D*BS/5 )/3 + C*D*AS*AS/5 )
            IF ( -HK .LT. 100 ) THEN
               B = SQRT(BS)
               BVN = BVN - EXP( -HK/2 )*SQRT(TWOPI)*PHID(-B/A)*B
     &                    *( 1 - C*BS*( 1 - D*BS/5 )/3 )
            ENDIF
            A = A/2
            DO I = 1, LG
               DO IS = -1, 1, 2
                  XS = ( A*(  IS*X(I,NG) + 1 ) )**2
                  RS = SQRT( 1 - XS )
                  ASR = -( BS/XS + HK )/2
                  IF ( ASR .GT. -100 ) THEN
                     BVN = BVN + A*W(I,NG)*EXP( ASR )
     &                    *( EXP( -HK*XS/( 2*( 1 + RS )**2 ) )/RS
     &                    - ( 1 + C*XS*( 1 + D*XS ) ) )
                  END IF
               END DO
            END DO
            BVN = -BVN/TWOPI
         ENDIF
         IF ( R .GT. 0 ) THEN
            BVN =  BVN + PHID( -MAX( H, K ) )
         ELSE
            BVN = -BVN
            IF ( K .GT. H ) THEN
               IF ( H .LT. 0 ) THEN
                  BVN = BVN + PHID(K)  - PHID(H)
               ELSE
                  BVN = BVN + PHID(-H) - PHID(-K)
               ENDIF
            ENDIF
         ENDIF
      ENDIF
      BVND = BVN
      END

