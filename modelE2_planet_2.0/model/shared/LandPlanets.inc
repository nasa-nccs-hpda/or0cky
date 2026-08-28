  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP001 = &
       PlanetParams_t( &
       name = 'LP001', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1948.9d0,   &  !surface pressure (mbar)
       ptop = 299.06900406504065, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9846d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.246554, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.39992d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP002 = &
       PlanetParams_t( &
       name = 'LP002', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1917.5d0,   &  !surface pressure (mbar)
       ptop = 294.250508130081301, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9898d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.163302, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40057d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP003 = &
       PlanetParams_t( &
       name = 'LP003', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 2289d0,   &  !surface pressure (mbar)
       ptop = 351.259146341463415, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.8675d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 30.121325, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.38596d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP004 = &
       PlanetParams_t( &
       name = 'LP004', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 987.99d0,   &  !surface pressure (mbar)
       ptop = 151.612286585365854, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9567d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.693233, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.3965d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP005 = &
       PlanetParams_t( &
       name = 'LP005', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1558.4d0,   &  !surface pressure (mbar)
       ptop = 239.144715447154472, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP006 = &
       PlanetParams_t( &
       name = 'LP006', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1676.1d0,   &  !surface pressure (mbar)
       ptop = 257.20640243902439, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.5366d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 35.419034, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.35165d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP007 = &
       PlanetParams_t( &
       name = 'LP007', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1012.4d0,   &  !surface pressure (mbar)
       ptop = 155.358130081300813, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9994d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.009606, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40177d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP008 = &
       PlanetParams_t( &
       name = 'LP008', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 3831.8d0,   &  !surface pressure (mbar)
       ptop = 588.009959349593496, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.4547d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 36.730253, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.34415d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP009 = &
       PlanetParams_t( &
       name = 'LP009', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1878.1d0,   &  !surface pressure (mbar)
       ptop = 288.204369918699187, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9903d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.155297, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40063d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP010 = &
       PlanetParams_t( &
       name = 'LP010', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1264d0,   &  !surface pressure (mbar)
       ptop = 193.967479674796748, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.7655d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 31.754345, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.37462d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP011 = &
       PlanetParams_t( &
       name = 'LP011', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 3032.8d0,   &  !surface pressure (mbar)
       ptop = 465.399186991869919, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.5016d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 35.979384, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.3484d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP012 = &
       PlanetParams_t( &
       name = 'LP012', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 3084.9d0,   &  !surface pressure (mbar)
       ptop = 473.394207317073171, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.2106d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 40.638294, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.32376d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP013 = &
       PlanetParams_t( &
       name = 'LP013', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 866.55d0,   &  !surface pressure (mbar)
       ptop = 132.976676829268293, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9715d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.456285, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.39831d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP014 = &
       PlanetParams_t( &
       name = 'LP014', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1896.6d0,   &  !surface pressure (mbar)
       ptop = 291.043292682926829, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.5606d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 35.034794, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.35391d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP015 = &
       PlanetParams_t( &
       name = 'LP015', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1270.9d0,   &  !surface pressure (mbar)
       ptop = 195.026321138211382, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP016 = &
       PlanetParams_t( &
       name = 'LP016', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1486.4d0,   &  !surface pressure (mbar)
       ptop = 228.095934959349594, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9191d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 29.295209, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.39198d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP017 = &
       PlanetParams_t( &
       name = 'LP017', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1675.6d0,   &  !surface pressure (mbar)
       ptop = 257.129674796747967, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9952d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.076848, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40124d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP018 = &
       PlanetParams_t( &
       name = 'LP018', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 540.73d0,   &  !surface pressure (mbar)
       ptop = 82.9778760162601626, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9988d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.019212, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40169d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP019 = &
       PlanetParams_t( &
       name = 'LP019', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1590.3d0,   &  !surface pressure (mbar)
       ptop = 244.039939024390244, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP020 = &
       PlanetParams_t( &
       name = 'LP020', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1683.7d0,   &  !surface pressure (mbar)
       ptop = 258.372662601626016, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9589d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.658011, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.39677d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP021 = &
       PlanetParams_t( &
       name = 'LP021', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 4984.7d0,   &  !surface pressure (mbar)
       ptop = 764.928556910569106, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.1035d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 42.352965, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.31562d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP022 = &
       PlanetParams_t( &
       name = 'LP022', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 958.19d0,   &  !surface pressure (mbar)
       ptop = 147.039319105691057, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9954d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.073646, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40126d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP023 = &
       PlanetParams_t( &
       name = 'LP023', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1240.6d0,   &  !surface pressure (mbar)
       ptop = 190.376626016260163, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP024 = &
       PlanetParams_t( &
       name = 'LP024', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1157.9d0,   &  !surface pressure (mbar)
       ptop = 177.685873983739837, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9997d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.004803, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.4018d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP025 = &
       PlanetParams_t( &
       name = 'LP025', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1887.4d0,   &  !surface pressure (mbar)
       ptop = 289.63150406504065, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP026 = &
       PlanetParams_t( &
       name = 'LP026', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 984.22d0,   &  !surface pressure (mbar)
       ptop = 151.033760162601626, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP027 = &
       PlanetParams_t( &
       name = 'LP027', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1917.6d0,   &  !surface pressure (mbar)
       ptop = 294.265853658536585, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP028 = &
       PlanetParams_t( &
       name = 'LP028', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1097.4d0,   &  !surface pressure (mbar)
       ptop = 168.401829268292683, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9075d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 29.480925, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.39061d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP029 = &
       PlanetParams_t( &
       name = 'LP029', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1684.6d0,   &  !surface pressure (mbar)
       ptop = 258.510772357723577, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP030 = &
       PlanetParams_t( &
       name = 'LP030', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1810.4d0,   &  !surface pressure (mbar)
       ptop = 277.815447154471545, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP031 = &
       PlanetParams_t( &
       name = 'LP031', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1259.8d0,   &  !surface pressure (mbar)
       ptop = 193.322967479674797, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP032 = &
       PlanetParams_t( &
       name = 'LP032', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 4451d0,   &  !surface pressure (mbar)
       ptop = 683.029471544715447, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.2094d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 40.657506, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.32366d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP033 = &
       PlanetParams_t( &
       name = 'LP033', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1030.9d0,   &  !surface pressure (mbar)
       ptop = 158.197052845528455, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP034 = &
       PlanetParams_t( &
       name = 'LP034', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1418.7d0,   &  !surface pressure (mbar)
       ptop = 217.707012195121951, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9965d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.056035, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40141d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP035 = &
       PlanetParams_t( &
       name = 'LP035', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 755.38d0,   &  !surface pressure (mbar)
       ptop = 115.917052845528455, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9872d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.204928, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40024d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP036 = &
       PlanetParams_t( &
       name = 'LP036', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1967.3d0,   &  !surface pressure (mbar)
       ptop = 301.892581300813008, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9994d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.009606, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40176d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP037 = &
       PlanetParams_t( &
       name = 'LP037', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 2011.4d0,   &  !surface pressure (mbar)
       ptop = 308.659959349593496, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9269d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 29.170331, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.3929d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP038 = &
       PlanetParams_t( &
       name = 'LP038', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 765.23d0,   &  !surface pressure (mbar)
       ptop = 117.428587398373984, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9999d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.001601, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP039 = &
       PlanetParams_t( &
       name = 'LP039', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 911.8d0,   &  !surface pressure (mbar)
       ptop = 139.920528455284553, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP040 = &
       PlanetParams_t( &
       name = 'LP040', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1530.4d0,   &  !surface pressure (mbar)
       ptop = 234.847967479674797, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9999d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.001601, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40183d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP041 = &
       PlanetParams_t( &
       name = 'LP041', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1307.3d0,   &  !surface pressure (mbar)
       ptop = 200.612093495934959, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.8964d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 29.658636, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.38931d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP042 = &
       PlanetParams_t( &
       name = 'LP042', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1757.8d0,   &  !surface pressure (mbar)
       ptop = 269.74369918699187, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP043 = &
       PlanetParams_t( &
       name = 'LP043', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1739.4d0,   &  !surface pressure (mbar)
       ptop = 266.920121951219512, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP044 = &
       PlanetParams_t( &
       name = 'LP044', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1384d0,   &  !surface pressure (mbar)
       ptop = 212.382113821138211, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9739d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.417861, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.39861d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP045 = &
       PlanetParams_t( &
       name = 'LP045', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1357.6d0,   &  !surface pressure (mbar)
       ptop = 208.330894308943089, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.958d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.67242, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.39666d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP046 = &
       PlanetParams_t( &
       name = 'LP046', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 7678.2d0,   &  !surface pressure (mbar)
       ptop = 1178.26036585365854, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.1929d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 40.921671, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.32238d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP047 = &
       PlanetParams_t( &
       name = 'LP047', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1449.7d0,   &  !surface pressure (mbar)
       ptop = 222.464126016260163, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9999d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.001601, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40183d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP048 = &
       PlanetParams_t( &
       name = 'LP048', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 569.16d0,   &  !surface pressure (mbar)
       ptop = 87.340609756097561, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9874d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.201726, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40027d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP049 = &
       PlanetParams_t( &
       name = 'LP049', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1113.2d0,   &  !surface pressure (mbar)
       ptop = 170.826422764227642, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.7756d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 31.592644, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.37571d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP050 = &
       PlanetParams_t( &
       name = 'LP050', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 733.11d0,   &  !surface pressure (mbar)
       ptop = 112.499603658536585, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9999d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.001601, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40182d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP051 = &
       PlanetParams_t( &
       name = 'LP051', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 500.01d0,   &  !surface pressure (mbar)
       ptop = 76.7291768292682927, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP052 = &
       PlanetParams_t( &
       name = 'LP052', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1623d0,   &  !surface pressure (mbar)
       ptop = 249.057926829268293, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP053 = &
       PlanetParams_t( &
       name = 'LP053', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 2551.6d0,   &  !surface pressure (mbar)
       ptop = 391.55650406504065, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.2372d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 40.212428, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.32585d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP054 = &
       PlanetParams_t( &
       name = 'LP054', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 706.85d0,   &  !surface pressure (mbar)
       ptop = 108.469867886178862, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9995d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.008005, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40178d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP055 = &
       PlanetParams_t( &
       name = 'LP055', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1488.6d0,   &  !surface pressure (mbar)
       ptop = 228.433536585365854, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.7671d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 31.728729, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.37479d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP056 = &
       PlanetParams_t( &
       name = 'LP056', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1395.3d0,   &  !surface pressure (mbar)
       ptop = 214.116158536585366, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP057 = &
       PlanetParams_t( &
       name = 'LP057', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1239.7d0,   &  !surface pressure (mbar)
       ptop = 190.238516260162602, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.5426d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 35.322974, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.35221d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP058 = &
       PlanetParams_t( &
       name = 'LP058', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1643.5d0,   &  !surface pressure (mbar)
       ptop = 252.203760162601626, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9992d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.012808, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40174d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP059 = &
       PlanetParams_t( &
       name = 'LP059', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1625.7d0,   &  !surface pressure (mbar)
       ptop = 249.472256097560976, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.8669d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 30.130931, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.38589d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP060 = &
       PlanetParams_t( &
       name = 'LP060', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1314.7d0,   &  !surface pressure (mbar)
       ptop = 201.747662601626016, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9983d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.027217, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40163d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP061 = &
       PlanetParams_t( &
       name = 'LP061', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1217.7d0,   &  !surface pressure (mbar)
       ptop = 186.8625, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9992d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.012808, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40174d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP062 = &
       PlanetParams_t( &
       name = 'LP062', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 10000d0,   &  !surface pressure (mbar)
       ptop = 1534.55284552845528, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.05d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 43.2095, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.31172d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP063 = &
       PlanetParams_t( &
       name = 'LP063', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 2000.1d0,   &  !surface pressure (mbar)
       ptop = 306.925914634146341, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP064 = &
       PlanetParams_t( &
       name = 'LP064', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1807.8d0,   &  !surface pressure (mbar)
       ptop = 277.416463414634146, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9938d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.099262, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40106d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP065 = &
       PlanetParams_t( &
       name = 'LP065', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1928.1d0,   &  !surface pressure (mbar)
       ptop = 295.877134146341463, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.7456d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 32.072944, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.3725d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP066 = &
       PlanetParams_t( &
       name = 'LP066', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1836.6d0,   &  !surface pressure (mbar)
       ptop = 281.835975609756098, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP067 = &
       PlanetParams_t( &
       name = 'LP067', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 693.27d0,   &  !surface pressure (mbar)
       ptop = 106.38594512195122, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP068 = &
       PlanetParams_t( &
       name = 'LP068', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1463.3d0,   &  !surface pressure (mbar)
       ptop = 224.551117886178862, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9999d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.001601, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40183d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP069 = &
       PlanetParams_t( &
       name = 'LP069', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 8427.7d0,   &  !surface pressure (mbar)
       ptop = 1293.27510162601626, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.1344d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 41.858256, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.31792d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP070 = &
       PlanetParams_t( &
       name = 'LP070', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1494.4d0,   &  !surface pressure (mbar)
       ptop = 229.323577235772358, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9997d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.004803, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40181d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP071 = &
       PlanetParams_t( &
       name = 'LP071', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 2477.9d0,   &  !surface pressure (mbar)
       ptop = 380.246849593495935, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.4524d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 36.767076, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.34395d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP072 = &
       PlanetParams_t( &
       name = 'LP072', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1229.3d0,   &  !surface pressure (mbar)
       ptop = 188.642581300813008, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9985d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.024015, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40165d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP073 = &
       PlanetParams_t( &
       name = 'LP073', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 835.31d0,   &  !surface pressure (mbar)
       ptop = 128.182733739837398, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.999d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.01601, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40172d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP074 = &
       PlanetParams_t( &
       name = 'LP074', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 5372.4d0,   &  !surface pressure (mbar)
       ptop = 824.423170731707317, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.3145d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 38.974855, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.3321d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP075 = &
       PlanetParams_t( &
       name = 'LP075', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 882.66d0,   &  !surface pressure (mbar)
       ptop = 135.448841463414634, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP076 = &
       PlanetParams_t( &
       name = 'LP076', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1104.1d0,   &  !surface pressure (mbar)
       ptop = 169.429979674796748, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP077 = &
       PlanetParams_t( &
       name = 'LP077', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 801.83d0,   &  !surface pressure (mbar)
       ptop = 123.04505081300813, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9807d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.308993, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.39944d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP078 = &
       PlanetParams_t( &
       name = 'LP078', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1823.1d0,   &  !surface pressure (mbar)
       ptop = 279.764329268292683, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9999d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.001601, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40182d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP079 = &
       PlanetParams_t( &
       name = 'LP079', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1917.6d0,   &  !surface pressure (mbar)
       ptop = 294.265853658536585, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.3597d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 38.251203, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.33589d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP080 = &
       PlanetParams_t( &
       name = 'LP080', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 662.35d0,   &  !surface pressure (mbar)
       ptop = 101.641107723577236, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP081 = &
       PlanetParams_t( &
       name = 'LP081', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1711d0,   &  !surface pressure (mbar)
       ptop = 262.561991869918699, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.999d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.01601, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40172d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP082 = &
       PlanetParams_t( &
       name = 'LP082', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 777.44d0,   &  !surface pressure (mbar)
       ptop = 119.302276422764228, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9947d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.084853, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40117d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP083 = &
       PlanetParams_t( &
       name = 'LP083', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 4261.4d0,   &  !surface pressure (mbar)
       ptop = 653.934349593495935, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.3016d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 39.181384, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.33104d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP084 = &
       PlanetParams_t( &
       name = 'LP084', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1120.8d0,   &  !surface pressure (mbar)
       ptop = 171.992682926829268, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9209d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 29.266391, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.3922d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP085 = &
       PlanetParams_t( &
       name = 'LP085', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 6291.9d0,   &  !surface pressure (mbar)
       ptop = 965.525304878048781, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.2116d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 40.622284, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.32383d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP086 = &
       PlanetParams_t( &
       name = 'LP086', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1957.9d0,   &  !surface pressure (mbar)
       ptop = 300.45010162601626, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP087 = &
       PlanetParams_t( &
       name = 'LP087', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1938.8d0,   &  !surface pressure (mbar)
       ptop = 297.519105691056911, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP088 = &
       PlanetParams_t( &
       name = 'LP088', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1583.2d0,   &  !surface pressure (mbar)
       ptop = 242.950406504065041, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.3736d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 38.028664, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.33707d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP089 = &
       PlanetParams_t( &
       name = 'LP089', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 804.21d0,   &  !surface pressure (mbar)
       ptop = 123.410274390243902, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP090 = &
       PlanetParams_t( &
       name = 'LP090', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 571.66d0,   &  !surface pressure (mbar)
       ptop = 87.7242479674796748, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9999d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.001601, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40183d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP091 = &
       PlanetParams_t( &
       name = 'LP091', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1245.9d0,   &  !surface pressure (mbar)
       ptop = 191.189939024390244, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9498d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.803702, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.39565d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP092 = &
       PlanetParams_t( &
       name = 'LP092', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1680.6d0,   &  !surface pressure (mbar)
       ptop = 257.896951219512195, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.986d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.22414, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.4001d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP093 = &
       PlanetParams_t( &
       name = 'LP093', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 980.51d0,   &  !surface pressure (mbar)
       ptop = 150.464441056910569, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.5583d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 35.071617, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.35369d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP094 = &
       PlanetParams_t( &
       name = 'LP094', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1496.2d0,   &  !surface pressure (mbar)
       ptop = 229.59979674796748, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9997d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.004803, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.4018d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP095 = &
       PlanetParams_t( &
       name = 'LP095', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1770.7d0,   &  !surface pressure (mbar)
       ptop = 271.723272357723577, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9999d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.001601, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40183d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP096 = &
       PlanetParams_t( &
       name = 'LP096', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 875.42d0,   &  !surface pressure (mbar)
       ptop = 134.337825203252033, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9929d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.113671, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40096d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP097 = &
       PlanetParams_t( &
       name = 'LP097', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 611.67d0,   &  !surface pressure (mbar)
       ptop = 93.8639939024390244, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9997d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.004803, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.4018d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP098 = &
       PlanetParams_t( &
       name = 'LP098', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 859.69d0,   &  !surface pressure (mbar)
       ptop = 131.923973577235772, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9471d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.846929, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.39533d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP099 = &
       PlanetParams_t( &
       name = 'LP099', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1206.5d0,   &  !surface pressure (mbar)
       ptop = 185.14380081300813, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9973d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.043227, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40151d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP100 = &
       PlanetParams_t( &
       name = 'LP100', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 2038.3d0,   &  !surface pressure (mbar)
       ptop = 312.787906504065041, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.6773d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 33.166427, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.36539d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP101 = &
       PlanetParams_t( &
       name = 'LP101', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1789.9d0,   &  !surface pressure (mbar)
       ptop = 274.669613821138211, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9999d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.001601, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40183d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP102 = &
       PlanetParams_t( &
       name = 'LP102', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1053.3d0,   &  !surface pressure (mbar)
       ptop = 161.634451219512195, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP103 = &
       PlanetParams_t( &
       name = 'LP103', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1099.9d0,   &  !surface pressure (mbar)
       ptop = 168.785467479674797, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9998d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.003202, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40181d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP104 = &
       PlanetParams_t( &
       name = 'LP104', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 9020.2d0,   &  !surface pressure (mbar)
       ptop = 1384.19735772357724, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.06904d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 42.9046696, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.3131d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP105 = &
       PlanetParams_t( &
       name = 'LP105', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1729.8d0,   &  !surface pressure (mbar)
       ptop = 265.446951219512195, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9985d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.024015, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40165d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP106 = &
       PlanetParams_t( &
       name = 'LP106', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 737.82d0,   &  !surface pressure (mbar)
       ptop = 113.222378048780488, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9833d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.267367, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.39976d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP107 = &
       PlanetParams_t( &
       name = 'LP107', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1599.6d0,   &  !surface pressure (mbar)
       ptop = 245.467073170731707, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP108 = &
       PlanetParams_t( &
       name = 'LP108', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1083.7d0,   &  !surface pressure (mbar)
       ptop = 166.299491869918699, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9974d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.041626, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40152d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP109 = &
       PlanetParams_t( &
       name = 'LP109', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1508.7d0,   &  !surface pressure (mbar)
       ptop = 231.517987804878049, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.89d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 29.7611, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.38855d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP110 = &
       PlanetParams_t( &
       name = 'LP110', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1549.6d0,   &  !surface pressure (mbar)
       ptop = 237.794308943089431, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9993d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.011207, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40175d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP001n = &
       PlanetParams_t( &
       name = 'LP001n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 9485.4d0,   &  !surface pressure (mbar)
       ptop = 1455.58475609756098, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9968d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.051232, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40144d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP002n = &
       PlanetParams_t( &
       name = 'LP002n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 9353.7d0,   &  !surface pressure (mbar)
       ptop = 1435.37469512195122, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9979d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.033621, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40158d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP003n = &
       PlanetParams_t( &
       name = 'LP003n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 9910.1d0,   &  !surface pressure (mbar)
       ptop = 1520.75721544715447, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9694d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.489906, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.39805d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP004n = &
       PlanetParams_t( &
       name = 'LP004n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 3319.9d0,   &  !surface pressure (mbar)
       ptop = 509.45619918699187, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9871d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.206529, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40024d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP005n = &
       PlanetParams_t( &
       name = 'LP005n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 7202.9d0,   &  !surface pressure (mbar)
       ptop = 1105.32306910569106, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP006n = &
       PlanetParams_t( &
       name = 'LP006n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 3029.6d0,   &  !surface pressure (mbar)
       ptop = 464.908130081300813, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.7436d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 32.104964, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.37229d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP007n = &
       PlanetParams_t( &
       name = 'LP007n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 3741.5d0,   &  !surface pressure (mbar)
       ptop = 574.152947154471545, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9998d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.003202, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40182d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP008n = &
       PlanetParams_t( &
       name = 'LP008n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 8369.3d0,   &  !surface pressure (mbar)
       ptop = 1284.31331300813008, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.7504d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 31.996096, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.373d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP009n = &
       PlanetParams_t( &
       name = 'LP009n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 9112.6d0,   &  !surface pressure (mbar)
       ptop = 1398.37662601626016, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.998d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.03202, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40159d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP010n = &
       PlanetParams_t( &
       name = 'LP010n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 3461.2d0,   &  !surface pressure (mbar)
       ptop = 531.139430894308943, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9144d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 29.370456, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.39142d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP011n = &
       PlanetParams_t( &
       name = 'LP011n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 6967.9d0,   &  !surface pressure (mbar)
       ptop = 1069.26107723577236, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.7831d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 31.472569, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.37652d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP012n = &
       PlanetParams_t( &
       name = 'LP012n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 3884.1d0,   &  !surface pressure (mbar)
       ptop = 596.035670731707317, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.3731d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 38.036669, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.33702d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP013n = &
       PlanetParams_t( &
       name = 'LP013n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 2665.3d0,   &  !surface pressure (mbar)
       ptop = 409.004369918699187, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9907d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.148893, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40069d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP014n = &
       PlanetParams_t( &
       name = 'LP014n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 4067.3d0,   &  !surface pressure (mbar)
       ptop = 624.148678861788618, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.7951d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 31.280449, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.37784d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP015n = &
       PlanetParams_t( &
       name = 'LP015n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 5382.5d0,   &  !surface pressure (mbar)
       ptop = 825.973069105691057, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP016n = &
       PlanetParams_t( &
       name = 'LP016n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 5985.8d0,   &  !surface pressure (mbar)
       ptop = 918.552642276422764, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9799d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.321801, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.39934d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP017n = &
       PlanetParams_t( &
       name = 'LP017n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 7894.9d0,   &  !surface pressure (mbar)
       ptop = 1211.51412601626016, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.999d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.01601, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40171d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP018n = &
       PlanetParams_t( &
       name = 'LP018n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 753.79d0,   &  !surface pressure (mbar)
       ptop = 115.673058943089431, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9991d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.014409, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40173d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP019n = &
       PlanetParams_t( &
       name = 'LP019n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 7405.5d0,   &  !surface pressure (mbar)
       ptop = 1136.41310975609756, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP020n = &
       PlanetParams_t( &
       name = 'LP020n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 7559.3d0,   &  !surface pressure (mbar)
       ptop = 1160.0145325203252, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9909d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.145691, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.4007d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP021n = &
       PlanetParams_t( &
       name = 'LP021n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 5070.1d0,   &  !surface pressure (mbar)
       ptop = 778.033638211382114, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.1186d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 42.111214, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.31674d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP022n = &
       PlanetParams_t( &
       name = 'LP022n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 3373.8d0,   &  !surface pressure (mbar)
       ptop = 517.727439024390244, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9987d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.020813, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40168d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP023n = &
       PlanetParams_t( &
       name = 'LP023n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 5190.3d0,   &  !surface pressure (mbar)
       ptop = 796.478963414634146, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP024n = &
       PlanetParams_t( &
       name = 'LP024n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 4664.7d0,   &  !surface pressure (mbar)
       ptop = 715.822865853658537, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9999d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.001601, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40183d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP025n = &
       PlanetParams_t( &
       name = 'LP025n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 9287.1d0,   &  !surface pressure (mbar)
       ptop = 1425.15457317073171, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP026n = &
       PlanetParams_t( &
       name = 'LP026n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 3566.7d0,   &  !surface pressure (mbar)
       ptop = 547.328963414634146, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP027n = &
       PlanetParams_t( &
       name = 'LP027n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 9477.6d0,   &  !surface pressure (mbar)
       ptop = 1454.38780487804878, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP028n = &
       PlanetParams_t( &
       name = 'LP028n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 3641.2d0,   &  !surface pressure (mbar)
       ptop = 558.761382113821138, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9721d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.446679, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.39838d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP029n = &
       PlanetParams_t( &
       name = 'LP029n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 8002.8d0,   &  !surface pressure (mbar)
       ptop = 1228.0719512195122, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP030n = &
       PlanetParams_t( &
       name = 'LP030n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 8799.1d0,   &  !surface pressure (mbar)
       ptop = 1350.26839430894309, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP031n = &
       PlanetParams_t( &
       name = 'LP031n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 5311.8d0,   &  !surface pressure (mbar)
       ptop = 815.123780487804878, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP032n = &
       PlanetParams_t( &
       name = 'LP032n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 3518.9d0,   &  !surface pressure (mbar)
       ptop = 539.99380081300813, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 44.01, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.30817d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP033n = &
       PlanetParams_t( &
       name = 'LP033n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 3862.6d0,   &  !surface pressure (mbar)
       ptop = 592.736382113821138, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP034n = &
       PlanetParams_t( &
       name = 'LP034n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 6287.2d0,   &  !surface pressure (mbar)
       ptop = 964.804065040650407, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9992d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.012808, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40174d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP035n = &
       PlanetParams_t( &
       name = 'LP035n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 2056.1d0,   &  !surface pressure (mbar)
       ptop = 315.519410569105691, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9953d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.075247, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40125d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP036n = &
       PlanetParams_t( &
       name = 'LP036n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 9785.1d0,   &  !surface pressure (mbar)
       ptop = 1501.57530487804878, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9999d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.001601, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40183d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP037n = &
       PlanetParams_t( &
       name = 'LP037n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 9140.9d0,   &  !surface pressure (mbar)
       ptop = 1402.71941056910569, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9839d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.257761, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.39984d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP038n = &
       PlanetParams_t( &
       name = 'LP038n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 2179.5d0,   &  !surface pressure (mbar)
       ptop = 334.455792682926829, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP039n = &
       PlanetParams_t( &
       name = 'LP039n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 3107.9d0,   &  !surface pressure (mbar)
       ptop = 476.923678861788618, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP040n = &
       PlanetParams_t( &
       name = 'LP040n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 7025.3d0,   &  !surface pressure (mbar)
       ptop = 1078.06941056910569, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP041n = &
       PlanetParams_t( &
       name = 'LP041n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 4755.7d0,   &  !surface pressure (mbar)
       ptop = 729.78729674796748, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9715d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.456285, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.39831d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP042n = &
       PlanetParams_t( &
       name = 'LP042n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 8466.4d0,   &  !surface pressure (mbar)
       ptop = 1299.21382113821138, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP043n = &
       PlanetParams_t( &
       name = 'LP043n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 8349.3d0,   &  !surface pressure (mbar)
       ptop = 1281.24420731707317, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP044n = &
       PlanetParams_t( &
       name = 'LP044n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 5870d0,   &  !surface pressure (mbar)
       ptop = 900.782520325203252, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9939d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.097661, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40107d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP045n = &
       PlanetParams_t( &
       name = 'LP045n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 5570.2d0,   &  !surface pressure (mbar)
       ptop = 854.776626016260163, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9898d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.163302, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40056d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP046n = &
       PlanetParams_t( &
       name = 'LP046n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 6712.3d0,   &  !surface pressure (mbar)
       ptop = 1030.03790650406504, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.07672d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 42.7817128, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.31365d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP047n = &
       PlanetParams_t( &
       name = 'LP047n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 6514d0,   &  !surface pressure (mbar)
       ptop = 999.607723577235772, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP048n = &
       PlanetParams_t( &
       name = 'LP048n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 892.61d0,   &  !surface pressure (mbar)
       ptop = 136.975721544715447, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.992d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.12808, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40084d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP049n = &
       PlanetParams_t( &
       name = 'LP049n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 2801.8d0,   &  !surface pressure (mbar)
       ptop = 429.951016260162602, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9108d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 29.428092, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.391d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP050n = &
       PlanetParams_t( &
       name = 'LP050n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1975.7d0,   &  !surface pressure (mbar)
       ptop = 303.181605691056911, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9999d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.001601, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP051n = &
       PlanetParams_t( &
       name = 'LP051n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 500d0,   &  !surface pressure (mbar)
       ptop = 76.7276422764227642, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP052n = &
       PlanetParams_t( &
       name = 'LP052n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 7612.6d0,   &  !surface pressure (mbar)
       ptop = 1168.19369918699187, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP053n = &
       PlanetParams_t( &
       name = 'LP053n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 3112.6d0,   &  !surface pressure (mbar)
       ptop = 477.644918699186992, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.3747d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 38.011053, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.33716d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP054n = &
       PlanetParams_t( &
       name = 'LP054n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1807.8d0,   &  !surface pressure (mbar)
       ptop = 277.416463414634146, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9998d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.003202, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40182d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP055n = &
       PlanetParams_t( &
       name = 'LP055n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 4564.8d0,   &  !surface pressure (mbar)
       ptop = 700.492682926829268, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.924d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 29.21676, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.39256d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP056n = &
       PlanetParams_t( &
       name = 'LP056n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 6169.9d0,   &  !surface pressure (mbar)
       ptop = 946.803760162601626, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP057n = &
       PlanetParams_t( &
       name = 'LP057n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1593.5d0,   &  !surface pressure (mbar)
       ptop = 244.53099593495935, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.6442d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 33.696358, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.36205d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP058n = &
       PlanetParams_t( &
       name = 'LP058n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 7733d0,   &  !surface pressure (mbar)
       ptop = 1186.66971544715447, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9998d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.003202, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40182d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP059n = &
       PlanetParams_t( &
       name = 'LP059n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 6259.8d0,   &  !surface pressure (mbar)
       ptop = 960.599390243902439, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9654d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.553946, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.39756d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP060n = &
       PlanetParams_t( &
       name = 'LP060n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 5645.9d0,   &  !surface pressure (mbar)
       ptop = 866.393191056910569, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9996d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.006404, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40179d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP061n = &
       PlanetParams_t( &
       name = 'LP061n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 5039.4d0,   &  !surface pressure (mbar)
       ptop = 773.322560975609756, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9998d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.003202, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40182d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP062n = &
       PlanetParams_t( &
       name = 'LP062n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 10000d0,   &  !surface pressure (mbar)
       ptop = 1534.55284552845528, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 44.01, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.30817d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP063n = &
       PlanetParams_t( &
       name = 'LP063n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 10000d0,   &  !surface pressure (mbar)
       ptop = 1534.55284552845528, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP064n = &
       PlanetParams_t( &
       name = 'LP064n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 8711.4d0,   &  !surface pressure (mbar)
       ptop = 1336.81036585365854, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9987d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.020813, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40168d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP065n = &
       PlanetParams_t( &
       name = 'LP065n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 6438.1d0,   &  !surface pressure (mbar)
       ptop = 987.960467479674797, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9238d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 29.219962, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.39254d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP066n = &
       PlanetParams_t( &
       name = 'LP066n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 8965d0,   &  !surface pressure (mbar)
       ptop = 1375.72662601626016, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP067n = &
       PlanetParams_t( &
       name = 'LP067n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1723.9d0,   &  !surface pressure (mbar)
       ptop = 264.541565040650407, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP068n = &
       PlanetParams_t( &
       name = 'LP068n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 6600.4d0,   &  !surface pressure (mbar)
       ptop = 1012.86626016260163, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP069n = &
       PlanetParams_t( &
       name = 'LP069n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 7295.1d0,   &  !surface pressure (mbar)
       ptop = 1119.47164634146341, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 44.01, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.30817d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP070n = &
       PlanetParams_t( &
       name = 'LP070n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 6795.1d0,   &  !surface pressure (mbar)
       ptop = 1042.74400406504065, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9999d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.001601, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40183d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP071n = &
       PlanetParams_t( &
       name = 'LP071n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 4433.5d0,   &  !surface pressure (mbar)
       ptop = 680.34400406504065, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.694d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 32.89906, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.3671d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP072n = &
       PlanetParams_t( &
       name = 'LP072n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 5106.9d0,   &  !surface pressure (mbar)
       ptop = 783.680792682926829, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9996d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.006404, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.4018d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP073n = &
       PlanetParams_t( &
       name = 'LP073n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 2618.5d0,   &  !surface pressure (mbar)
       ptop = 401.822662601626016, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9997d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.004803, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.4018d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP074n = &
       PlanetParams_t( &
       name = 'LP074n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 8033.6d0,   &  !surface pressure (mbar)
       ptop = 1232.79837398373984, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.5416d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 35.338984, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.35211d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP075n = &
       PlanetParams_t( &
       name = 'LP075n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 2923.3d0,   &  !surface pressure (mbar)
       ptop = 448.595833333333333, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP076n = &
       PlanetParams_t( &
       name = 'LP076n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 4325.1d0,   &  !surface pressure (mbar)
       ptop = 663.709451219512195, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP077n = &
       PlanetParams_t( &
       name = 'LP077n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 2313.7d0,   &  !surface pressure (mbar)
       ptop = 355.049491869918699, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9933d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.107267, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40101d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP078n = &
       PlanetParams_t( &
       name = 'LP078n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 8877.9d0,   &  !surface pressure (mbar)
       ptop = 1362.36067073170732, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP079n = &
       PlanetParams_t( &
       name = 'LP079n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1702.2d0,   &  !surface pressure (mbar)
       ptop = 261.211585365853659, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.2787d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 39.548013, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.32917d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP080n = &
       PlanetParams_t( &
       name = 'LP080n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1528.1d0,   &  !surface pressure (mbar)
       ptop = 234.495020325203252, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP081n = &
       PlanetParams_t( &
       name = 'LP081n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 8159.5d0,   &  !surface pressure (mbar)
       ptop = 1252.11839430894309, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9998d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.003202, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40182d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP082n = &
       PlanetParams_t( &
       name = 'LP082n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 2230.8d0,   &  !surface pressure (mbar)
       ptop = 342.328048780487805, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9981d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.030419, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40161d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP083n = &
       PlanetParams_t( &
       name = 'LP083n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 5472.2d0,   &  !surface pressure (mbar)
       ptop = 839.738008130081301, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.4561d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 36.707839, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.34428d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP084n = &
       PlanetParams_t( &
       name = 'LP084n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 3870.9d0,   &  !surface pressure (mbar)
       ptop = 594.010060975609756, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9771d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.366629, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.399d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP085n = &
       PlanetParams_t( &
       name = 'LP085n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 5764.4d0,   &  !surface pressure (mbar)
       ptop = 884.577642276422764, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.1394d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 41.778206, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.3183d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP086n = &
       PlanetParams_t( &
       name = 'LP086n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 9733.2d0,   &  !surface pressure (mbar)
       ptop = 1493.6109756097561, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP087n = &
       PlanetParams_t( &
       name = 'LP087n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 9612.4d0,   &  !surface pressure (mbar)
       ptop = 1475.07357723577236, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP088n = &
       PlanetParams_t( &
       name = 'LP088n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1079.3d0,   &  !surface pressure (mbar)
       ptop = 165.624288617886179, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.08115d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 42.7107885, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.31398d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP089n = &
       PlanetParams_t( &
       name = 'LP089n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 2426.7d0,   &  !surface pressure (mbar)
       ptop = 372.389939024390244, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP090n = &
       PlanetParams_t( &
       name = 'LP090n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 953.41d0,   &  !surface pressure (mbar)
       ptop = 146.305802845528455, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9999d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.001601, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40183d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP091n = &
       PlanetParams_t( &
       name = 'LP091n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 4827.4d0,   &  !surface pressure (mbar)
       ptop = 740.790040650406504, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.987d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.20813, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40022d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP092n = &
       PlanetParams_t( &
       name = 'LP092n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 7828.5d0,   &  !surface pressure (mbar)
       ptop = 1201.32469512195122, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.997d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.04803, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40147d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP093n = &
       PlanetParams_t( &
       name = 'LP093n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 800.2d0,   &  !surface pressure (mbar)
       ptop = 122.794918699186992, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.4587d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 36.666213, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.34451d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP094n = &
       PlanetParams_t( &
       name = 'LP094n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 6805.8d0,   &  !surface pressure (mbar)
       ptop = 1044.3859756097561, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9999d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.001601, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40183d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP095n = &
       PlanetParams_t( &
       name = 'LP095n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 8547.3d0,   &  !surface pressure (mbar)
       ptop = 1311.62835365853659, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP096n = &
       PlanetParams_t( &
       name = 'LP096n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 2838.3d0,   &  !surface pressure (mbar)
       ptop = 435.552134146341463, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9978d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.035222, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40157d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP097n = &
       PlanetParams_t( &
       name = 'LP097n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1205.9d0,   &  !surface pressure (mbar)
       ptop = 185.051727642276423, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9998d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.003202, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40182d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP098n = &
       PlanetParams_t( &
       name = 'LP098n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 2489.9d0,   &  !surface pressure (mbar)
       ptop = 382.088313008130081, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9817d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.292983, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.39957d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP099n = &
       PlanetParams_t( &
       name = 'LP099n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 4954.5d0,   &  !surface pressure (mbar)
       ptop = 760.294207317073171, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9993d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.011207, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40176d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP100n = &
       PlanetParams_t( &
       name = 'LP100n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 6076.2d0,   &  !surface pressure (mbar)
       ptop = 932.425, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.8917d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 29.733883, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.38876d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP101n = &
       PlanetParams_t( &
       name = 'LP101n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 8668d0,   &  !surface pressure (mbar)
       ptop = 1330.15040650406504, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP102n = &
       PlanetParams_t( &
       name = 'LP102n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 4004d0,   &  !surface pressure (mbar)
       ptop = 614.434959349593496, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP103n = &
       PlanetParams_t( &
       name = 'LP103n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 4297.7d0,   &  !surface pressure (mbar)
       ptop = 659.504776422764228, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9999d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.001601, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40183d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP104n = &
       PlanetParams_t( &
       name = 'LP104n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 8397.5d0,   &  !surface pressure (mbar)
       ptop = 1288.64075203252033, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 44.01, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.30817d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP105n = &
       PlanetParams_t( &
       name = 'LP105n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 8272.4d0,   &  !surface pressure (mbar)
       ptop = 1269.44349593495935, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9997d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.004803, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.4018d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP106n = &
       PlanetParams_t( &
       name = 'LP106n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 1928.2d0,   &  !surface pressure (mbar)
       ptop = 295.892479674796748, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9936d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.102464, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40104d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP107n = &
       PlanetParams_t( &
       name = 'LP107n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 7464.3d0,   &  !surface pressure (mbar)
       ptop = 1145.43628048780488, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 1d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28., &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40184d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP108n = &
       PlanetParams_t( &
       name = 'LP108n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 4179.2d0,   &  !surface pressure (mbar)
       ptop = 641.320325203252033, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9993d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.011207, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40176d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP109n = &
       PlanetParams_t( &
       name = 'LP109n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 5837.2d0,   &  !surface pressure (mbar)
       ptop = 895.749186991869919, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9716d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.454684, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.39831d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       LP110n = &
       PlanetParams_t( &
       name = 'LP110n', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 7140.2d0,   &  !surface pressure (mbar)
       ptop = 1095.70142276422764, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.9998d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.003202, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.40182d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

  type(PlanetParams_t), parameter, private :: & ! Land Planets, no TOPO, atm.composition: N2 and CO2
       EarthLikePlanet = &
       PlanetParams_t( &
       name = 'EarthLikePlanet', &
       grav = 9.80655, &       ! m s-2 (like Earth)
       radius = 6.3781d06, &   ! equatorial radius (m) (like Earth)
       psf = 984d0,   &  !surface pressure (mbar)
       ptop =     151.00, &  !top of sigma-levels pressure (mbar); has to be way above mountains; here ptop=psf*150/984
       pN2 = 0.78084d0,   &  !fraction N2              here: pCO2=1-pN2
       pO2 = 0.0d0, &  !fraction O2
       pAr = 0.0d0, &  !fraction Argon
       mair = 28.9655, &  !molecular weight of air, here: pN2*28.0134d0 + pCO2*44.01d0
       srat = 1.401d0  &  !cp/cv, cp,cv spec.heats, here:(cp_N2*pN2+cp_CO2*pCO2)/(cv_N2*pN2+cv_CO2*pCO2)
       )  ! note: cp and cv are temperature dependent, here we use the value at 0 degC

