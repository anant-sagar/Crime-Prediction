from pandas.core.frame import DataFrame
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle



st.title("Crime Prediction")

#st.image("images\crowdimage.jpg")


st.success('''Crime detection is one of the highly useful applications in the fields of deep learning, 
as this helps in curbing the crime and increasing the safety of people.''')



if st.checkbox("About"):
    st.markdown("""Data mining and machine learning have become a vital part of crime detection and prevention. 
    The purpose of this paper is to evaluate data mining methods and their performances that can be used for analyzing the
    collected data about the past crimes. I identified the most appropriate data mining methods to analyze the collected data
    from sources specialized in crime prevention by comparing them theoretically and practically. Some attributes of this dataset
    are, gender, age, employment status, crime place. Methods are applied on these data to determine their effectiveness in analyzing
    and preventing crime. Evaluations on the data showed that the method with a higher performance is “Decision Tree”. This was
    achieved by some performance measures, such as the number of instances correctly classified, accuracy or precision and recall,
    that has brought better results compared to other methods. I come to the conclusion that the data mining methods contribute to
    the predictions on the possibility of occurrence of the crime and as a result in its prevention.""")

if  st.checkbox("Make Prediction"):

    col11, col12 = st.beta_columns(2)

    with col11:

        population= st.number_input("Enter Population")

    with col12:
       householdsize =st.number_input("Enter House Hold Size")

    col1, col2 = st.beta_columns(2)

    with col1:
       racepctblack =st.number_input("Enter racepctblack")
    with col2:
        racePctWhite=st.number_input("Enter racePctWhite")

    col3, col4 = st.beta_columns(2)
    
    with col3:
        racePctAsian= st.number_input("Enter racePctAsian")

    with col4:
        racePctHisp= st.number_input("Enter racePctHisp")

    col5, col6 = st.beta_columns(2)
    
    with col5:
        agePct12t21= st.number_input("Enter agePct12t21")

    with col6:
        agePct12t29= st.number_input("Enter agePct12t29")

    col7, col8 = st.beta_columns(2)
    
    with col5:
        agePct16t24= st.number_input("Enter agePct16t24")

        

    with col6:
        agePct65up= st.number_input("Enter agePct65up")

    col9, col10 = st.beta_columns(2)
    
    with col9:
        numbUrban= st.number_input("Enter numbUrban")
        
    with col10:  
        pctUrban= st.number_input("Enter pctUrban")

    col13, col14 = st.beta_columns(2)
    
    with col13:
        medIncome= st.number_input("Enter medIncome")

    with col14:  
        pctWWage= st.number_input("Enter pctWWage")

    col15, col16 = st.beta_columns(2)
    
    with col15:
        pctWFarmSelf= st.number_input("Enter pctWFarmSelf")

    with col16:  
        pctWInvInc= st.number_input("Enter pctWInvInc")

    
    col17, col18 = st.beta_columns(2)
    
    with col17:
        pctWSocSec= st.number_input("Enter pctWSocSec")

    with col18:  
        pctWPubAsst= st.number_input("Enter pctWPubAsst")

    col19, col20 = st.beta_columns(2)
    
    with col19:
        pctWRetire= st.number_input("Enter pctWRetire")

    with col20:  
        medFamInc= st.number_input("Enter medFamInc")

    col21, col22 = st.beta_columns(2)
    
    with col21:
        perCapInc= st.number_input("Enter perCapInc")

    with col22:  
        whitePerCap= st.number_input("Enter whitePerCap")
    
  
    blackPerCap= st.number_input("Enter blackPerCap")

    col29, col30 = st.beta_columns(2)
    
    with col29:
        indianPerCap= st.number_input("Enter indianPerCap")

    with col30:  
        AsianPerCap= st.number_input("Enter AsianPerCap")
    
    col31, col32 = st.beta_columns(2)
    
    with col31:
        OtherPerCap= st.number_input("Enter OtherPerCap")

    with col32:  
        HispPerCap= st.number_input("Enter HispPerCap")
    
    col33, col34 = st.beta_columns(2)
    
    with col33:
        NumUnderPov= st.number_input("Enter NumUnderPov")

    with col34:  
        PctPopUnderPov= st.number_input("Enter PctPopUnderPov")
    
    col35, col36 = st.beta_columns(2)
    
    with col35:
        PctLess9thGrade= st.number_input("Enter PctLess9thGrade")

    with col36:  
        PctNotHSGrad= st.number_input("Enter PctNotHSGrad")
    
    col37, col38 = st.beta_columns(2)
    
    with col37:
        PctBSorMore= st.number_input("Enter PctBSorMore")

    with col38:  
        PctUnemployed= st.number_input("Enter PctUnemployed")
    
    col39, col40 = st.beta_columns(2)
    
    with col39:
        PctEmploy= st.number_input("Enter PctEmploy")

    with col40:  
        PctEmplManu= st.number_input("Enter PctEmplManu")
    
    col41, col42 = st.beta_columns(2)
    
    with col41:
        PctEmplProfServ= st.number_input("Enter PctEmplProfServ")

    with col42:  
        PctOccupManu= st.number_input("Enter PctOccupManu")
    
    col43, col44 = st.beta_columns(2)
    
    with col43:
        PctOccupMgmtProf= st.number_input("Enter PctOccupMgmtProf")

    with col44:  
        MalePctDivorce= st.number_input("Enter MalePctDivorce")
    
    col45, col46 = st.beta_columns(2)
    
    with col45:
        MalePctNevMarr= st.number_input("Enter MalePctNevMarr")

    with col46:  
        FemalePctDiv= st.number_input("Enter FemalePctDiv")
    
    col47, col48 = st.beta_columns(2)
    
    with col47:
        TotalPctDiv= st.number_input("Enter TotalPctDiv")

    with col48:  
        PersPerFam= st.number_input("Enter PersPerFam")
    
    col49, col50 = st.beta_columns(2)
    
    with col49:
        PctFam2Par= st.number_input("Enter PctFam2Par")

    with col50:  
        PctKids2Par= st.number_input("Enter PctKids2Par")
    
    col51, col52 = st.beta_columns(2)
    
    with col51:
        PctYoungKids2Par= st.number_input("Enter PctYoungKids2Par")

    with col52:  
        PctTeen2Par= st.number_input("Enter PctTeen2Par")
            
    col53, col54 = st.beta_columns(2)
    
    with col53:
        PctWorkMomYoungKids= st.number_input("Enter PctWorkMomYoungKids")

    with col54:  
        PctWorkMom= st.number_input("Enter PctWorkMom")
            
    col55, col56 = st.beta_columns(2)
    
    with col55:
        NumIlleg= st.number_input("Enter NumIlleg")

    with col56:  
        PctIlleg= st.number_input("Enter PctIlleg")
            
    col57, col58 = st.beta_columns(2)
    
    with col57:
        NumImmig= st.number_input("Enter NumImmig")

    with col58:  
        PctImmigRecent= st.number_input("Enter PctImmigRecent")
            
    col59, col60 = st.beta_columns(2)
    
    with col59:
        PctImmigRec5= st.number_input("Enter PctImmigRec5")

    with col60:  
        PctImmigRec8= st.number_input("Enter PctImmigRec8")
            
    col61, col62 = st.beta_columns(2)
    
    with col61:
        PctImmigRec10= st.number_input("Enter PctImmigRec10")

    with col62:  
        PctRecentImmig= st.number_input("Enter PctRecentImmig")
            
    col63, col64 = st.beta_columns(2)
    
    with col63:
        PctRecImmig5= st.number_input("Enter PctRecImmig5")

    with col64:  
        PctRecImmig8= st.number_input("Enter PctRecImmig8")
            
    col65, col66 = st.beta_columns(2)
    
    with col65:
        PctRecImmig10= st.number_input("Enter PctRecImmig10")

    with col66:  
        PctSpeakEnglOnly= st.number_input("Enter PctSpeakEnglOnly")
            
    col67, col68 = st.beta_columns(2)
    
    with col67:
        PctNotSpeakEnglWell= st.number_input("Enter PctNotSpeakEnglWell")

    with col68:  
        PctLargHouseFam= st.number_input("Enter PctLargHouseFam")
            
    col69, col70 = st.beta_columns(2)
    
    with col69:
        PctLargHouseOccup= st.number_input("Enter PctLargHouseOccup")

    with col70:  
        PersPerOccupHous= st.number_input("Enter PersPerOccupHous")
            
    col71, col72 = st.beta_columns(2)
    
    with col71:
        PersPerOwnOccHous= st.number_input("Enter PersPerOwnOccHous")

    with col72:  
        PersPerRentOccHous= st.number_input("Enter PersPerRentOccHous")
            
    col73, col74 = st.beta_columns(2)
    
    with col73:
        PctPersOwnOccup= st.number_input("Enter PctPersOwnOccup")

    with col74:  
        PctPersDenseHous= st.number_input("Enter PctPersDenseHous")
            
    col75, col76 = st.beta_columns(2)
    
    with col75:
        PctHousLess3BR= st.number_input("Enter PctHousLess3BR")

    with col76:  
        MedNumBR= st.number_input("Enter MedNumBR")
            
    col77, col78 = st.beta_columns(2)
    
    with col77:
        HousVacant= st.number_input("Enter HousVacant")

    with col78:  
        PctHousOccup= st.number_input("Enter PctHousOccup")
            
    col79, col80 = st.beta_columns(2)
    
    with col79:
        PctHousOwnOcc= st.number_input("Enter PctHousOwnOcc")

    with col80:  
        PctVacantBoarded= st.number_input("Enter PctVacantBoarded")
            
    col81, col82 = st.beta_columns(2)
    
    with col81:
        PctVacMore6Mos= st.number_input("Enter PctVacMore6Mos")

    with col82:  
        MedYrHousBuilt= st.number_input("Enter MedYrHousBuilt")
            
    col83, col84 = st.beta_columns(2)
    
    with col83:
        PctHousNoPhone= st.number_input("Enter PctHousNoPhone")

    with col84:  
        PctWOFullPlumb= st.number_input("Enter PctWOFullPlumb")
            
    col85, col86 = st.beta_columns(2)
    
    with col85:
        OwnOccLowQuart= st.number_input("Enter OwnOccLowQuart")

    with col86:  
        OwnOccMedVal= st.number_input("Enter OwnOccMedVal")
            
    col87, col88 = st.beta_columns(2)
    
    with col87:
        OwnOccHiQuart= st.number_input("Enter OwnOccHiQuart")

    with col88:  
        RentLowQ= st.number_input("Enter RentLowQ")
            
    col89, col90 = st.beta_columns(2)
    
    with col89:
        RentMedian= st.number_input("Enter RentMedian")

    with col90:  
        RentHighQ= st.number_input("Enter RentHighQ")
                    
    col91, col92 = st.beta_columns(2)
    
    with col91:
        MedRent= st.number_input("Enter MedRent")

    with col92:  
        MedRentPctHousInc= st.number_input("Enter MedRentPctHousInc")
                    
    col93, col94 = st.beta_columns(2)
    
    with col93:
        MedOwnCostPctInc= st.number_input("Enter MedOwnCostPctInc")

    with col94:  
        MedOwnCostPctIncNoMtg= st.number_input("Enter MedOwnCostPctIncNoMtg")
                    
    col95, col96 = st.beta_columns(2)
    
    with col95:
        NumInShelters= st.number_input("Enter NumInShelters")

    with col96:  
        NumStreet= st.number_input("Enter NumStreet")
                    
    col97, col98 = st.beta_columns(2)
    
    with col97:
        PctForeignBorn= st.number_input("Enter PctForeignBorn")

    with col98:  
        PctBornSameState= st.number_input("Enter PctBornSameState")
                    
    col99, col100 = st.beta_columns(2)
    
    with col99:
        PctSameHouse85= st.number_input("Enter PctSameHouse85")

    with col100:  
        PctSameCity85= st.number_input("Enter PctSameCity85")
                    
    col101, col102 = st.beta_columns(2)
    
    with col101:
        PctSameState85= st.number_input("Enter PctSameState85")

    with col102:  
        LandArea= st.number_input("Enter LandArea")
                    
    col103, col104 = st.beta_columns(2)
    
    with col103:
        PopDens= st.number_input("Enter PopDens")

    with col104:  
        PctUsePubTrans= st.number_input("Enter PctUsePubTrans")
                    
    LemasPctOfficDrugUn= st.number_input("Enter LemasPctOfficDrugUn")

if st.checkbox("Visualization"):
    visualization= st.sidebar.selectbox("Training Data Graphs",["agePct12t21_chart","agePct12t21_scatter","agePct12t29_chart","agePct12t29_scatter","agePct16t24_chart"
    ,"agePct16t24_scatter","agePct65up_chart","agePct65up_scatter","AsianPerCap_chart","AsianPerCap_scatter","blackPerCap_chart","blackPerCap_scatter",
    "FemalePctDiv_chart","FemalePctDiv_scatter","HispPerCap_chart","HispPerCap_scatter","highCrime_chart","highCrime_scatter","householdsize_chart","householdsize_scatter",
    "HousVacant_chart","HousVacant_scatter","indianPerCap_chart","indianPerCap_scatter","LandArea_chart","LandArea_scatter","LemasPctOfficDrugUn_chart",
    "LemasPctOfficDrugUn_scatter","MalePctDivorce_chart","MalePctDivorce_scatter","MalePctNevMarr_chart","MalePctNevMarr_scatter","medFamInc_chart",
    "medFamInc_scatter","medIncome_chart","medIncome_scatter","MedNumBR_chart","MedNumBR_sactter","MedOwnCostPctInc_chart","MedOwnCostPctInc_scatter",
    "MedOwnCostPctIncNoMtg_chart","MedOwnCostPctIncNoMtg_scatter","MedRent_chart","MedRent_sactter","MedRentPctHousInc_chart","MedRentPctHousInc_scatter",
    "MedYrHousBuilt_chart","MedYrHousBuilt_scatter","numbUrban_chart","numbUrban_scatter","NumImmig_chart","NumImmig_scatter","NumInShelters_chart","NumInShelters_scatter",
    "NumStreet_chart","NumStreet_scatter","NumStreet_chart","NumStreet_scatter","OtherPerCap_chart","OtherPerCap_scatter","OwnOccHiQuart_chart","OwnOccHiQuart_scatter",
    "OwnOccLowQuart_chart","OwnOccLowQuart_scatter","OwnOccMedVal_chart","OwnOccMedVal_scatter","PctBornSameState_chart","PctBornSameState_scatter","PctBSorMore_chart","PctBSorMore_scatter",
    "PctEmplManu_chart","PctEmplManu_scatter","PctEmploy_chart","PctEmploy_scatter","PctEmplProfServ_chart","PctEmplProfServ_scatter","PctFam2Par_chart","PctFam2Par_scatter","PctForeignBorn_chart","PctForeignBorn_scatter",
    "PctHousLess3BR_chart","PctHousLess3BR_scatter","PctHousNoPhone_chart","PctHousNoPhone_scatter","PctHousOccup_chart","PctHousOccup_scatter","PctHousOwnOcc_chart","PctHousOwnOcc_scatter","PctIlleg_chart","PctIlleg_scatter",
    "PctImmigRec5_chart","PctImmigRec5_scatter","PctImmigRec8_chart","PctImmigRec8_scatter","PctImmigRec10_chart","PctImmigRec10_scatter","PctImmigRecent_chart","PctImmigRecent_scatter",
    "PctKids2Par_chart","PctKids2Par_scatter","PctLargHouseFam_chart","PctLargHouseFam_scatter","PctLargHouseOccup_chart","PctLargHouseOccup_scatter","PctLess9thGrade_chart","PctLess9thGrade_scatter","PctNotHSGrad_chart","PctNotHSGrad_scatter",
    "PctNotSpeakEnglWell_chart","PctNotSpeakEnglWell_scatter","PctOccupManu_chart","PctOccupManu_scatter","PctOccupMgmtProf_chart","PctOccupMgmtProf_scatter","PctPersDenseHous_chart","PctPersDenseHous_scatter","PctPersOwnOccup_chart","PctPersOwnOccup_scatter",
    "PctPopUnderPov_chart","PctPopUnderPov_scatter","PctRecentImmig_chart","PctRecentImmig_scatter","PctRecImmig5_chart","PctRecImmig5_scatter","PctRecImmig8_chart","PctRecImmig8_scatter","PctRecImmig5_chart","PctRecImmig5_scatter",
    "PctRecImmig8_chart","PctRecImmig8_scatter","PctRecImmig10_chart","PctRecImmig10_scatter","PctSameCity85_chart","PctSameCity85_scatter","PctSameHouse85_chart","PctSameHouse85_scatter","PctSameState85_chart","PctSameState85_scatter","PctSpeakEnglOnly_chart","PctSpeakEnglOnly_scatter",
    "PctTeen2Par_chart","PctTeen2Par_scatter","PctUnemployed_chart","PctUnemployed_scatter","pctUrban_chart","pctUrban_sactter","PctUsePubTrans_chart","PctUsePubTrans_scatter","PctVacantBoarded_chart","PctVacantBoarded_scatter","PctVacMore6Mos_chart","PctVacMore6Mos_scatter","pctWFarmSelf_chart",
    "pctWFarmSelf_scatter","pctWInvInc_chart","pctWInvInc_scatter","PctWOFullPlumb_chart","PctWOFullPlumb_scatter","PctWorkMom_chart","PctWorkMom_scatter","PctWorkMomYoungKids_chart","PctWorkMomYoungKids_scatter","pctWPubAsst_chart","pctWPubAsst_scatter","pctWRetire_chart","pctWRetire_scatter","pctWSocSec_chart",
    "pctWSocSec_scatter","pctWWage_chart","pctWWage_scatter","PctYoungKids2Par_chart","PctYoungKids2Par_scatter","perCapInc_chart","perCapInc_scatter","PersPerFam_chart","PersPerFam_scatter","PersPerOccupHous_chart","PersPerOccupHous_scatter","PersPerOwnOccHous_chart","PersPerOwnOccHous_scatter","PersPerRentOccHous_chart",
    "PersPerRentOccHous_scatter","PopDens_chart","PopDen_scatter","population_chart","population_scatter","racePctAsian_chart","racePctAsian_scatter","racepctblack_chart","racepctblack_scatter","racePctHisp_chart","racePctHisp_scatter","racePctWhite_chart","racePctWhite_scatter",
    "RentHighQ_chart","RentHighQ_scatter","RentLowQ_chart","RentLowQ_scatter","RentMedian_chart","RentMedian_scatter","TotalPctDiv_chart","TotalPctDiv_scatter","ViolentCrimesPerPop_chart","ViolentCrimesPerPop_scatter","whitePerCap_chart","whitePerCap_scatter"])


    if  visualization=="agePct12t21_chart":
        st.image("img/agePct12t21_chart.png")

    if  visualization=="agePct12t21_scatter":
        st.image("img/agePct12t21_scatter.png")

    if  visualization=="agePct12t29_chart":
        st.image("img/agePct12t29_chart.png")

    if  visualization=="agePct12t29_scatter":
        st.image("img/agePct12t29_scatter.png")

    if  visualization=="agePct16t24_chart":
        st.image("img/agePct16t24_chart.png")

    if  visualization=="agePct16t24_scatter":
        st.image("img/agePct16t24_scatter.png")

    if  visualization=="agePct65up_chart":
        st.image("img/agePct65up_chart.png")

    if  visualization=="agePct65up_scatter":
        st.image("img/agePct65up_scatter.png")

    if  visualization=="AsianPerCap_chart":
        st.image("img/AsianPerCap_chart.png")

    if  visualization=="AsianPerCap_scatter":
        st.image("img/AsianPerCap_scatter.png")

    if  visualization=="blackPerCap_chart":
        st.image("img/blackPerCap_chart.png")

    if  visualization=="blackPerCap_scatter":
        st.image("img/blackPerCap_scatter.png")

    if  visualization=="FemalePctDiv_chart":
        st.image("img/FemalePctDiv_chart.png")

    if  visualization=="FemalePctDiv_scatter":
        st.image("img/FemalePctDiv_scatter.png")

    if  visualization=="highCrime_chart":
        st.image("img/highCrime_chart.png")

    if  visualization=="highCrime_scatter":
        st.image("img/highCrime_scatter.png")

    if  visualization=="HispPerCap_chart":
        st.image("img/HispPerCap_chart.png")

    if  visualization=="HispPerCap_scatter":
        st.image("img/HispPerCap_scatter.png")

    if  visualization=="householdsize_chart":
        st.image("img/householdsize_chart.png")

    if  visualization=="householdsize_scatter":
        st.image("img/householdsize_scatter.png")

    if  visualization=="HousVacant_chart":
        st.image("img/HousVacant_chart.png")

    if  visualization=="HousVacant_scatter":
        st.image("img/HousVacant_scatter.png")

    if  visualization=="indianPerCap_chart":
        st.image("img/indianPerCap_chart.png")

    if  visualization=="indianPerCap_scatter":
        st.image("img/indianPerCap_scatter.png")

    if  visualization=="LandArea_chart":
        st.image("img/LandArea_chart.png")

    if  visualization=="LandArea_scatter":
        st.image("img/LandArea_scatter.png")

    if  visualization=="LemasPctOfficDrugUn_chart":
        st.image("img/LemasPctOfficDrugUn_chart.png")

    if  visualization=="LemasPctOfficDrugUn_scatter":
        st.image("img/LemasPctOfficDrugUn_scatter.png")

    if  visualization=="MalePctDivorce_chart":
        st.image("img/MalePctDivorce_chart.png")

    if  visualization=="MalePctDivorce_scatter":
        st.image("img/MalePctDivorce_scatter.png")

    if  visualization=="MalePctNevMarr_chart":
        st.image("img/MalePctNevMarr_chart.png")

    if  visualization=="MalePctNevMarr_scatter":
        st.image("img/MalePctNevMarr_scatter.png")

    if  visualization=="medFamInc_chart":
        st.image("img/medFamInc_chart.png")

    if  visualization=="medFamInc_scatter":
        st.image("img/medFamInc_scatter.png")

    if  visualization=="medIncome_chart":
        st.image("img/medIncome_chart.png")

    if  visualization=="medIncome_scatter":
        st.image("img/medIncome_scatter.png")

    if  visualization=="MedNumBR_chart":
        st.image("img/MedNumBR_chart.png")

    if  visualization=="MedNumBR_chart":
        st.image("img/MedNumBR_chart.png")

    if  visualization=="MedOwnCostPctInc_chart":
        st.image("img/MedOwnCostPctInc_chart.png")

    if  visualization=="MedOwnCostPctInc_chart":
        st.image("img/MedOwnCostPctInc_chart.png")

    if  visualization=="MedOwnCostPctIncNoMtg_chart":
        st.image("img/MedOwnCostPctIncNoMtg_chart.png")

    if  visualization=="MedOwnCostPctIncNoMtg_scatter":
        st.image("img/MedOwnCostPctIncNoMtg_scatter.png")

    if  visualization=="MedRent_chart":
        st.image("img/MedRent_chart.png")

    if  visualization=="MedRent_sactter":
        st.image("img/MedRent_sactter.png")

    if  visualization=="MedRentPctHousInc_chart":
        st.image("img/MedRentPctHousInc_chart.png")

    if  visualization=="MedRentPctHousInc_scatter":
        st.image("img/MedRentPctHousInc_scatter.png")

    if  visualization=="MedYrHousBuilt_chart":
        st.image("img/MedYrHousBuilt_chart.png")

    if  visualization=="MedYrHousBuilt_scatter":
        st.image("img/MedYrHousBuilt_scatter.png")
        
    if  visualization=="numbUrban_chart":
        st.image("img/numbUrban_chart.png")

    if  visualization=="numbUrban_scatter":
        st.image("img/numbUrban_scatter.png")

    if  visualization=="NumIlleg_chart":
        st.image("img/NumIlleg_chart.png")

    if  visualization=="NumIlleg_scatter":
        st.image("img/NumIlleg_scatter.png")


    if  visualization=="NumImmig_chart":
        st.image("img/NumImmig_chart.png")

    if  visualization=="NumImmig_scatter":
        st.image("img/NumImmig_scatter.png")

    if  visualization=="NumInShelters_chart":
        st.image("img/NumInShelters_chart.png")

    if  visualization=="NumInShelters_scatter":
        st.image("img/NumInShelters_scatter.png")

    if  visualization=="NumStreet_chart":
        st.image("img/NumStreet_chart.png")

    if  visualization=="NumStreet_scatter":
        st.image("img/NumStreet_scatter.png")

    if  visualization=="NumUnderPov_chart":
        st.image("img/NumUnderPov_chart.png")

    if  visualization=="NumUnderPov_scatter":
        st.image("img/NumUnderPov_scatter.png")

    if  visualization=="OtherPerCap_chart":
        st.image("img/OtherPerCap_chart.png")

    if  visualization=="OtherPerCap_scatter":
        st.image("img/OtherPerCap_scatter.png")

    if  visualization=="OwnOccHiQuart_chart":
        st.image("img/OwnOccHiQuart_chart.png")

    if  visualization=="OwnOccHiQuart_scatter":
        st.image("img/OwnOccHiQuart_scatter.png")

    if  visualization=="OwnOccHiQuart_chart":
        st.image("img/OwnOccHiQuart_chart.png")

    if  visualization=="OwnOccHiQuart_scatter":
        st.image("img/OwnOccHiQuart_scatter.png")

    if  visualization=="OwnOccLowQuart_chart":
        st.image("img/OwnOccLowQuart_chart.png")

    if  visualization=="OwnOccLowQuart_scatter":
        st.image("img/OwnOccLowQuart_scatter.png")

    if  visualization=="OwnOccMedVal_chart":
        st.image("img/OwnOccMedVal_chart.png")

    if  visualization=="OwnOccMedVal_scatter":
        st.image("img/OwnOccMedVal_scatter.png")
        
    if  visualization=="PctBornSameState_chart":
        st.image("img/PctBornSameState_chart.png")

    if  visualization=="PctBornSameState_scatter":
        st.image("img/PctBornSameState_scatter.png")

    if  visualization=="PctBSorMore_chart":
        st.image("img/PctBSorMore_chart.png")

    if  visualization=="PctBSorMore_scatter":
        st.image("img/PctBSorMore_scatter.png")

    if  visualization=="PctEmplManu_chart":
        st.image("img/PctEmplManu_chart.png")

    if  visualization=="PctEmplManu_scatter":
        st.image("img/PctEmplManu_scatter.png")

    if  visualization=="PctEmploy_chart":
        st.image("img/PctEmploy_chart.png")

    if  visualization=="PctEmploy_scatter":
        st.image("img/PctEmploy_scatter.png")

    if  visualization=="PctEmplProfServ_chart":
        st.image("img/PctEmplProfServ_chart.png")

    if  visualization=="PctEmplProfServ_scatter":
        st.image("img/PctEmplProfServ_scatter.png")

    if  visualization=="PctFam2Par_chart":
        st.image("img/PctFam2Par_chart.png")

    if  visualization=="PctFam2Par_scatter":
        st.image("img/PctFam2Par_scatter.png")

    if  visualization=="PctForeignBorn_chart":
        st.image("img/PctForeignBorn_chart.png")

    if  visualization=="PctForeignBorn_scatter":
        st.image("img/PctForeignBorn_scatter.png")

    
    if  visualization=="PctHousLess3BR_chart":
        st.image("img/PctHousLess3BR_chart.png")

    if  visualization=="PctHousLess3BR_scatter":
        st.image("img/PctHousLess3BR_scatter.png")

    if  visualization=="PctHousNoPhone_chart":
        st.image("img/PctHousNoPhone_chart.png")

    if  visualization=="PctHousNoPhone_scatter":
        st.image("img/PctHousNoPhone_scatter.png")

    if  visualization=="PctHousOccup_chart":
        st.image("img/PctHousOccup_chart.png")

    if  visualization=="PctHousOccup_scatter":
        st.image("img/PctHousOccup_scatter.png")

    if  visualization=="PctHousOwnOcc_chart":
        st.image("img/PctHousOwnOcc_chart.png")

    if  visualization=="PctHousOwnOcc_scatter":
        st.image("img/PctHousOwnOcc_scatter.png")

    if  visualization=="PctIlleg_chart":
        st.image("img/PctIlleg_chart.png")

    if  visualization=="PctIlleg_scatter":
        st.image("img/PctIlleg_scatter.png")

    if  visualization=="PctImmigRec5_chart":
        st.image("img/PctImmigRec5_chart.png")

    if  visualization=="PctImmigRec5_chart":
        st.image("img/PctImmigRec5_chart.png")

    if  visualization=="PctImmigRec8_chart":
        st.image("img/PctImmigRec8_chart.png")

    if  visualization=="PctImmigRec8_scatter":
        st.image("img/PctImmigRec8_scatter.png")

    if  visualization=="PctImmigRec10_chart":
        st.image("img/PctImmigRec10_chart.png")

    if  visualization=="PctImmigRec10_scatter":
        st.image("img/PctImmigRec10_scatter.png")

    if  visualization=="PctImmigRecent_chart":
        st.image("img/PctImmigRecent_chart.png")

    if  visualization=="PctImmigRecent_scatter":
        st.image("img/PctImmigRecent_scatter.png")

    if  visualization=="PctKids2Par_chart":
        st.image("img/PctKids2Par_chart.png")

    if  visualization=="PctKids2Par_scatter":
        st.image("img/PctKids2Par_scatter.png")

    if  visualization=="PctLargHouseFam_chart":
        st.image("img/PctLargHouseFam_chart.png")

    if  visualization=="PctLargHouseFam_scatter":
        st.image("img/PctLargHouseFam_scatter.png")

    if  visualization=="PctLargHouseOccup_chart":
        st.image("img/PctLargHouseOccup_chart.png")

    if  visualization=="PctLargHouseOccup_scatter":
        st.image("img/PctLargHouseOccup_scatter.png")
        
    if  visualization=="PctLess9thGrade_chart":
        st.image("img/PctLess9thGrade_chart.png")

    if  visualization=="PctLess9thGrade_scatter":
        st.image("img/PctLess9thGrade_scatter.png")

    if  visualization=="PctNotHSGrad_chart":
        st.image("img/PctNotHSGrad_chart.png")

    if  visualization=="PctNotHSGrad_scatter":
        st.image("img/PctNotHSGrad_scatter.png")

    if  visualization=="PctNotSpeakEnglWell_chart":
        st.image("img/PctNotSpeakEnglWell_chart.png")

    if  visualization=="PctNotSpeakEnglWell_scatter":
        st.image("img/PctNotSpeakEnglWell_scatter.png")

    if  visualization=="PctOccupManu_chart":
        st.image("img/PctOccupManu_chart.png")

    if  visualization=="PctOccupManu_scatter":
        st.image("img/PctOccupManu_scatter.png")

    if  visualization=="PctOccupMgmtProf_chart":
        st.image("img/PctOccupMgmtProf_chart.png")

    if  visualization=="PctOccupMgmtProf_scatter":
        st.image("img/PctOccupMgmtProf_scatter.png")

    if  visualization=="PctPersDenseHous_chart":
        st.image("img/PctPersDenseHous_chart.png")

    if  visualization=="PctPersDenseHous_scatter":
        st.image("img/PctPersDenseHous_scatter.png")

    if  visualization=="PctPersOwnOccup_chart":
        st.image("img/PctPersOwnOccup_chart.png")

    if  visualization=="PctPersOwnOccup_scatter":
        st.image("img/PctPersOwnOccup_scatter.png")

    if  visualization=="PctPopUnderPov_chart":
        st.image("img/PctPopUnderPov_chart.png")

    if  visualization=="PctPopUnderPov_scatter":
        st.image("img/PctPopUnderPov_scatter.png")

    if  visualization=="PctRecentImmig_chart":
        st.image("img/PctRecentImmig_chart.png")

    if  visualization=="PctRecentImmig_scatter":
        st.image("img/PctRecentImmig_scatter.png")

    if  visualization=="PctRecImmig5_chart":
        st.image("img/PctRecImmig5_chart.png")

    if  visualization=="PctRecImmig5_scatter":
        st.image("img/PctRecImmig5_scatter.png")

    if  visualization=="PctRecImmig8_chart":
        st.image("img/PctRecImmig8_chart.png")

    if  visualization=="PctRecImmig8_scatter":
        st.image("img/PctRecImmig8_scatter.png")

    if  visualization=="PctRecImmig10_chart":
        st.image("img/PctRecImmig10_chart.png")

    if  visualization=="PctRecImmig10_scatter":
        st.image("img/PctRecImmig10_scatter.png")

    if  visualization=="PctSameCity85_chart":
        st.image("img/PctSameCity85_chart.png")

    if  visualization=="PctSameCity85_scatter":
        st.image("img/PctSameCity85_scatter.png")


    if  visualization=="PctSameHouse85_chart":
        st.image("img/PctSameHouse85_chart.png")

    if  visualization=="PctSameHouse85_scatter":
        st.image("img/PctSameHouse85_scatter.png")

    if  visualization=="PctSameState85_chart":
        st.image("img/PctSameState85_chart.png")

    if  visualization=="PctSameState85_scatter":
        st.image("img/PctSameState85_scatter.png")

    if  visualization=="PctSpeakEnglOnly_chart":
        st.image("img/PctSpeakEnglOnly_chart.png")

    if  visualization=="PctSpeakEnglOnly_scatter":
        st.image("img/PctSpeakEnglOnly_scatter.png")

    if  visualization=="PctTeen2Par_chart":
        st.image("img/PctTeen2Par_chart.png")

    if  visualization=="PctTeen2Par_scatter":
        st.image("img/PctTeen2Par_scatter.png")

    if  visualization=="PctUnemployed_chart":
        st.image("img/PctUnemployed_chart.png")

    if  visualization=="PctUnemployed_scatter":
        st.image("img/PctUnemployed_scatter.png")

    if  visualization=="pctUrban_chart":
        st.image("img/pctUrban_chart.png")

    if  visualization=="pctUrban_scatter":
        st.image("img/pctUrban_scatter.png")

    if  visualization=="PctUsePubTrans_chart":
        st.image("img/PctUsePubTrans_chart.png")

    if  visualization=="PctUsePubTrans_scatter":
        st.image("img/PctUsePubTrans_scatter.png")

    if  visualization=="PctVacantBoarded_chart":
        st.image("img/PctVacantBoarded_chart.png")

    if  visualization=="PctVacantBoarded_scatter":
        st.image("img/PctVacantBoarded_scatter.png")

    if  visualization=="PctVacMore6Mos_chart":
        st.image("img/PctVacMore6Mos_chart.png")

    if  visualization=="PctVacMore6Mos_scatter":
        st.image("img/PctVacMore6Mos_scatter.png")

    if  visualization=="pctWFarmSelf_chart":
        st.image("img/pctWFarmSelf_chart.png")

    if  visualization=="pctWFarmSelf_scatter":
        st.image("img/pctWFarmSelf_scatter.png")

    if  visualization=="pctWInvInc_chart":
        st.image("img/pctWInvInc_chart.png")

    if  visualization=="pctWInvInc_scatter":
        st.image("img/pctWInvInc_scatter.png")

    if  visualization=="PctWOFullPlumb_chart":
        st.image("img/PctWOFullPlumb_chart.png")

    if  visualization=="PctWOFullPlumb_scatter":
        st.image("img/PctWOFullPlumb_scatter.png")

    if  visualization=="PctWorkMom_chart":
        st.image("img/PctWorkMom_chart.png")

    if  visualization=="PctWorkMom_scatter":
        st.image("img/PctWorkMom_scatter.png")

    if  visualization=="PctWorkMomYoungKids_chart":
        st.image("img/PctWorkMomYoungKids_chart.png")

    if  visualization=="PctWorkMomYoungKids_scatter":
        st.image("img/PctWorkMomYoungKids_scatter.png")

    if  visualization=="pctWPubAsst_chart":
        st.image("img/pctWPubAsst_chart.png")

    
    if  visualization=="pctWPubAsst_scatter":
        st.image("img/pctWPubAsst_scatter.png")

    if  visualization=="pctWRetire_chart":
        st.image("img/pctWRetire_chart.png")

    if  visualization=="pctWRetire_scatter":
        st.image("img/pctWRetire_scatter.png")

    if  visualization=="pctWSocSec_chart":
        st.image("img/pctWSocSec_chart.png")

    if  visualization=="pctWSocSec_scatter":
        st.image("img/pctWSocSec_scatter.png")

    if  visualization=="pctWWage_chart":
        st.image("img/pctWWage_chart.png")

    if  visualization=="pctWWage_scatter":
        st.image("img/pctWWage_scatter.png")

    if  visualization=="PctYoungKids2Par_chart":
        st.image("img/PctYoungKids2Par_chart.png")

    if  visualization=="PctYoungKids2Par_scatter":
        st.image("img/PctYoungKids2Par_scatter.png")

    if  visualization=="perCapInc_chart":
        st.image("img/perCapInc_chart.png")

    if  visualization=="perCapInc_scatter":
        st.image("img/perCapInc_scatter.png")

    if  visualization=="PersPerFam_chart":
        st.image("img/PersPerFam_chart.png")

    if  visualization=="PersPerFam_scatter":
        st.image("img/PersPerFam_scatter.png")

    if  visualization=="PersPerOccupHous_chart":
        st.image("img/PersPerOccupHous_chart.png")

    if  visualization=="PersPerOccupHous_scatter":
        st.image("img/PersPerOccupHous_scatter.png")

    if  visualization=="PersPerOwnOccHous_chart":
        st.image("img/PersPerOwnOccHous_chart.png")

    if  visualization=="PersPerOwnOccHous_scatter":
        st.image("img/PersPerOwnOccHous_scatter.png")

    if  visualization=="PersPerRentOccHous_chart":
        st.image("img/PersPerRentOccHous_chart.png")

    if  visualization=="PersPerRentOccHous_scatter":
        st.image("img/PersPerRentOccHous_scatter.png")

    if  visualization=="PopDens_chart":
        st.image("img/PopDens_chart.png")

    if  visualization=="PopDens_scatter":
        st.image("img/PopDens_scatter.png")

    if  visualization=="population_chart":
        st.image("img/population_chart.png")

    if  visualization=="population_scatter":
        st.image("img/population_scatter.png")

    if  visualization=="racePctAsian_chart":
        st.image("img/racePctAsian_chart.png")

    if  visualization=="racePctAsian_scatter":
        st.image("img/racePctAsian_scatter.png")

    if  visualization=="racepctblack_chart":
        st.image("img/racepctblack_chart.png")

    if  visualization=="racepctblack_scatter":
        st.image("img/racepctblack_scatter.png")

    if  visualization=="racePctHisp_chart":
        st.image("img/racePctHisp_chart.png")

    if  visualization=="racePctHisp_scatter":
        st.image("img/racePctHisp_scatter.png")

    if  visualization=="racePctWhite_chart":
        st.image("img/racePctWhite_chart.png")

    if  visualization=="racePctWhite_scatter":
        st.image("img/racePctWhite_scatter.png")

    if  visualization=="RentHighQ_chart":
        st.image("img/RentHighQ_chart.png")

    if  visualization=="RentHighQ_scatter":
        st.image("img/RentHighQ_scatter.png")

    if  visualization=="RentLowQ_chart":
        st.image("img/RentLowQ_chart.png")

    if  visualization=="RentLowQ_scatter":
        st.image("img/RentLowQ_scatter.png")

    if  visualization=="RentMedian_chart":
        st.image("img/RentMedian_chart.png")

    if  visualization=="RentMedian_scatter":
        st.image("img/RentMedian_scatter.png")

    if  visualization=="TotalPctDiv_chart":
        st.image("img/TotalPctDiv_chart.png")

    if  visualization=="TotalPctDiv_scatter":
        st.image("img/TotalPctDiv_scatter.png")

    if  visualization=="ViolentCrimesPerPop_chart":
        st.image("img/ViolentCrimesPerPop_chart.png")

    if  visualization=="ViolentCrimesPerPop_scatter":
        st.image("img/ViolentCrimesPerPop_scatter.png")

    if  visualization=="whitePerCap_chart":
        st.image("img/whitePerCap_chart.png")

    if  visualization=="whitePerCap_scatter":
        st.image("img/whitePerCap_scatter.png")