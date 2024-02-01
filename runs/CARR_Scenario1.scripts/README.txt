
README

Prior to running the scripts below, execute the following:
    
  mkdir runs && cd runs
  wget -r -np -nH -R "index.html*" --cut-dirs=2 http://mag.gmu.edu/git-data/ccmc/CARR_Scenario1/

Edit CARR_Scenarios1_info.py, change the value of data_dir to the "runs" directory
created above.

Execute the files in the following order:
    
    1) CARR_Scenario1_2Dplots.py
    2) CARR_Scenario1_2Dnedplots.py
    3) CARR_Scenario1_Findboundaries.py
    4) CARR_Scenario1_Findboundaries_part2.py
    5) CARR_Scenario1_deltaB_by_region.py
    6) CARR_Scenario1_heatmaps.py
    7) CARR_Scenario1_solar_wind.py

Before proceeding, goto to the Chigomezyo_Ngwira_09112_3a.scripts folder, read the 
README, and execute the scripts in Chigomezyo_Ngwira_092112_3a.scripts.

After executing the Chigomezyo_Ngwira_09112_3a scripts, edit
CARR_Chigo_compare_heatmaps.py in the CARR_Scenario1.scripts directory.
Change the value of data_dir to the "runs" directory containing the CARR_Scenario1 
and the Chigomezyo_Ngwira_09112_3a data. Then execute:
    
    1) CARR_Chigo_compare_heatmaps.py

