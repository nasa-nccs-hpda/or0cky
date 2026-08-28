#!/bin/ksh
# Example script to take monthly OMA (TCADI) taijl scaled output from the
# ./$IN/ dir and create timestream gas chemistry and aerosol input for use
# in a run that will use coupled_chem=0 parameter. (This example acts upon
# already-averaged decadal climatology files from YYY0 to YYYY9 with
# timestream representative years YYY4, since 10-year averages don't have
# a middle year. This was chosen to match current NINT constituent input
# Oct 2, 2018).


#### USER SETTINGS ####

NCO=/usr/local/other/SLES11.1/nco/4.4.4/intel-12.1.0.233/bin/
run=E14TomaOCNf10_4av
diag=taijl
# Measure time from January of year yref. Used 1750 to match emissions and
# current NINT constituent input:
yref=1750
IN=DATA
OUT=NEW
d1=185 ; d2=200 # to do 1850s to 2000s

chemDir=$OUT/uncoupledChem_${run}_decadal_F40
aeroDir=$OUT/uncoupledAero_${run}_decadal_F40


#### FUNCTIONS ####

checkUnits() {
  foundUnits=$( ncdump -h $1 | grep ${2}: | grep units | cut -d\" -f 2 )
  if [[ $foundUnits != $3 ]] ; then
    echo "Unexpected units found for ${2}: ${foundUnits}. Exiting."
    exit 1
  fi
}

concatenateInTime() {
  # add a time variable that counts months from Jan of yref...
  timeNow=$( echo " ( $3 - $4 ) * 12 + $5 " | bc )
  echo "TIME = $timeNow for $2"
  $NCO/ncap2 -A -s "defdim(\"time\",1) ; time[time]=${timeNow}." $1 $1
  # ... and give units to time:
  $NCO/ncap2 -A -s "time@units=\"months since ${4}-01\";" $1 $1
  # ... and make that dimension the unlimited record dimension:
  $NCO/ncks -O --mk_rec time $1 $1
  # ... adding time dimension to the variables (see
  # https://sourceforge.net/p/nco/discussion/9830/thread/0851525d/):
  $NCO/ncwa -a time $1 __${1}
  $NCO/ncecat -O -u time __${1} __${1}
  mv __${1} $1
  # ... add to the ongoing file:
  if [[ -e $2 ]] ; then
    mv ${2} __temp
    $NCO/ncrcat __temp $1 $2
  else
    mv $1 $2
  fi
  [[ -e __temp ]] && rm __temp
}

extractRenameConvert() {
    $NCO/ncks -A -v $3 $1 $2
    if [[ $3 != $4 ]] ; then
      $NCO/ncrename -v ${3},${4} $2
    fi
    if [[ $5 != 1.0 ]] ; then
      $NCO/ncap2 -A -s "${4}=${4}*${5};" $2 $2
    fi
    $NCO/ncatted -a units,${4},o,c,"${6}" $2
}


#### MAIN ####

# prepare output dirs:
[[ ! -d $chemDir ]] && mkdir -p $chemDir
[[ ! -d $aeroDir ]] && mkdir -p $aeroDir

set -A month JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC
# begin looping over decades:
x=$d1
while [[ ${x} -le $d2 ]] ; do
  y=${x}0-${x}9
  yr=${x}4
  chemFile=${chemDir}/${yr}.nc
  aeroFile=${aeroDir}/${yr}.nc
  if [[ -e $chemFile || -e $aeroFile ]] ; then
    echo "one or more file exists that should not:"
    echo $chemFile $aeroFile
    exit 2
  fi

  # and over months:
  m=0
  while [[ ${m} -le 11 ]]; do
    f=${IN}/${month[${m}]}${y}.${diag}${run}.nc
    vf=vars_$( basename $f)
    [[ -e $vf ]] && rm $vf



    # CHEM --> AEROSOLS:

    # Prepare the ozone, converting to mole O3 / mole air:
    vin=O3_vmr ; expectedUnits='ppbv'
    checkUnits $f $vin "$expectedUnits"
    vout=o3_offline ; newUnits='mole O3 / mole air' ; conv=1.e-9
    extractRenameConvert $f $vf $vin $vout $conv "$newUnits"
    $NCO/ncap2 -A -s "${vout}@long_name=\"O3 volume mixing ratio from run ${run}\";" $vf $vf

    # Prepare the OH concentration, converting to molecules cm-3:
    vin=OH_conc ; expectedUnits='10^5 molecules cm-3'
    checkUnits $f $vin "$expectedUnits"
    vout=ohr ; newUnits='molecules cm-3' ; conv=1.e5
    extractRenameConvert $f $vf $vin $vout $conv "$newUnits"
    $NCO/ncap2 -A -s "${vout}@long_name=\"OH concentration from run ${run}\";" $vf $vf

    # Prepare the HO2 concentration, converting to molecules cm-3:
    vin=HO2_conc ; expectedUnits='10^7 molecules cm-3'
    checkUnits $f $vin "$expectedUnits"
    vout=dho2r ; newUnits='molecules cm-3' ; conv=1.e7
    extractRenameConvert $f $vf $vin $vout $conv "$newUnits"
    $NCO/ncap2 -A -s "${vout}@long_name=\"HO2 concentration from run ${run}\";" $vf $vf

    # Prepare the H2O2 photolysis rate, converting to s-1:
    vin=JH2O2 ; expectedUnits='10^2 s-1'
    checkUnits $f $vin "$expectedUnits"
    vout=perjr ; newUnits='s-1' ; conv=1.e2
    extractRenameConvert $f $vf $vin $vout $conv "$newUnits"
    $NCO/ncap2 -A -s "${vout}@long_name=\"H2O2 photolysis rate from run ${run}\";" $vf $vf

    # Prepare the NO3 concentration, converting to molecules cm-3:
    vin=NO3_conc ; expectedUnits='10^5 molecules cm-3'
    checkUnits $f $vin "$expectedUnits"
    vout=tno3r ; newUnits='molecules cm-3' ; conv=1.e5
    extractRenameConvert $f $vf $vin $vout $conv "$newUnits"
    $NCO/ncap2 -A -s "${vout}@long_name=\"NO3 concentration from run ${run}\";" $vf $vf

    # Prepare the HNO3, converting to mole HNO3 / mole air:
    vin=HNO3 ; expectedUnits='10^-10 V/V air'
    checkUnits $f $vin "$expectedUnits"
    vout=off_HNO3 ; newUnits='mole HNO3 / mole air' ; conv=1.e-10
    extractRenameConvert $f $vf $vin $vout $conv "$newUnits"
    $NCO/ncap2 -A -s "${vout}@long_name=\"HNO3 volume mixing ratio from run ${run}\";" $vf $vf

    concatenateInTime $vf $chemFile $yr $yref $m
    [[ -e $vf ]] && rm $vf


    # AEROSOLS --> CHEM:

    # Prepare the DMS, converting to mole DMS / mole air:
    vin=DMS ; expectedUnits='10^-11 V/V air'
    checkUnits $f $vin "$expectedUnits"
    vout=${vin} ; newUnits="mole ${vout} / mole air" ; conv=1.e-11
    extractRenameConvert $f $vf $vin $vout $conv "$newUnits"
    $NCO/ncap2 -A -s "${vout}@long_name=\"${vout} volume mixing ratio from run ${run}\";" $vf $vf

    # Prepare the SO2, converting to mole SO2 / mole air:
    vin=SO2 ; expectedUnits='10^-10 V/V air'
    checkUnits $f $vin "$expectedUnits"
    vout=${vin} ; newUnits="mole ${vout} / mole air" ; conv=1.e-10
    extractRenameConvert $f $vf $vin $vout $conv "$newUnits"
    $NCO/ncap2 -A -s "${vout}@long_name=\"${vout} volume mixing ratio from run ${run}\";" $vf $vf

    # Prepare the SO4, converting to kg SO4 / kg air
    vin=SO4 ; expectedUnits='10^-10 kg/kg air'
    checkUnits $f $vin "$expectedUnits"
    vout=${vin} ; newUnits="kg ${vout} / kg air" ; conv=1.e-10
    extractRenameConvert $f $vf $vin $vout $conv "$newUnits"
    $NCO/ncap2 -A -s "${vout}@long_name=\"${vout} mass mixing ratio from run ${run}\";" $vf $vf

    concatenateInTime $vf $aeroFile $yr $yref $m
    [[ -e $vf ]] && rm $vf

    let m+=1
  done # months
  let x+=1
done # decades

