# Import required libraries
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="EPQ",
    page_icon="âï¸",
    layout="wide",
)

st.sidebar.header("EPQ model")

st.markdown("""
            ### âï¸ Tool for identifying optimal lotsize.
            ð¨âð» Michal ÄurÄÃ­k\\
            ð 8.2.2023
            """)

@st.cache
def convert_df(df):
    return df.to_csv().encode("utf-8")       
         
@st.cache
def EPQ(metrika):
    if metrika == "setup costs":
        new_df = pd.DataFrame(columns=("yearly demand", "production rate","setup costs", "holding costs","Q" , "T", "Lots", "Tp", "Td", "APC", "AHC", "Imax", "ATC"))
        for vyrobok, row in raw_data.iterrows():
            D = raw_data.loc[vyrobok, "yearly demand"]
            P = raw_data.loc[vyrobok, "production rate"]
            S = raw_data.loc[vyrobok, "setup costs"]
            C = raw_data.loc[vyrobok, "holding costs"]
            
            I = options/100
            
            if(D>0 and P>0 and S>0 and C>0):
                
                H = (1+I)*C
                Q = (np.sqrt((2*D*S)/(H*(1-(D/P)))))
                T = round(Q/D*12,2)
                number_of_production_runs = D/Q
                Tp = round(Q/P*12,2)
                Td = round((Q/D-Q/P)*12,2)
                APC = number_of_production_runs*S
                AHC = Q/2*(1-(D/P))*H
                Imax = Q*(1-(D/P))
                ATC = APC+AHC
                new_df.loc[vyrobok] = [D, P, S, C, Q , T, number_of_production_runs, Tp, Td, APC, AHC, Imax, ATC]
            
            else:
                print("All the input data must be positive !!!")
        
        return new_df
    else:
        new_df = pd.DataFrame(columns=("yearly demand", "production rate","alternative costs", "holding costs","Q" , "T", "Lots", "Tp", "Td", "APC", "AHC", "Imax", "ATC"))
        for vyrobok, row in raw_data.iterrows():
            D = raw_data.loc[vyrobok, "yearly demand"]
            P = raw_data.loc[vyrobok, "production rate"]
            S = raw_data.loc[vyrobok, "alternative costs"]
            C = raw_data.loc[vyrobok, "holding costs"]
            
            I = options/100
            
            if(D>0 and P>0 and S>0 and C>0):
                
                H = (1+I)*C
                Q = (np.sqrt((2*D*S)/(H*(1-(D/P)))))
                T = round(Q/D*12,2)
                number_of_production_runs = D/Q
                Tp = round(Q/P*12,2)
                Td = round((Q/D-Q/P)*12,2)
                APC = number_of_production_runs*S
                AHC = (Q/2)*(1-(D/P))*H
                Imax = Q*(1-(D/P))
                ATC = APC+AHC
                new_df.loc[vyrobok] = [D, P, S, C, Q , T, number_of_production_runs, Tp, Td, APC, AHC, Imax, ATC]
            
            else:
                print("All the input data must be positive !!!")
        
        return new_df

def show_product (product, data):
    data_graph = pd.DataFrame(columns = ("Holding costs", "Setup costs", "Total costs","Lot size", "Lots"))
    Q = data.loc[product, "Q"]
    D = data.loc[product, "yearly demand"]
    P = data.loc[product, "production rate"]
    if "setup costs" in data.columns:
        S = data.loc[product, "setup costs"]
    else:
        S = data.loc[product, "alternative costs"]
    C = data.loc[product, "holding costs"]
    
    I = options/100
    
    for lotsize in np.linspace(Q*0.5, Q*1.5,100):
        data_graph.loc[len(data_graph)] = [((lotsize)/2)*(1-(D/P))*((1+I)*C), (D/lotsize)*S, (((lotsize)/2)*(1-(D/P))*((1+I)*C))+ ((D/lotsize)*S), lotsize, D/lotsize ]
    
    fig = px.line(data_graph, y = ["Holding costs", "Setup costs", "Total costs"], x = "Lot size", labels={
                     "variable": "Legend",
                     "value": "Costs (â¬)",
                     "index": "Lots"
                 })
    return fig , data_graph

def inv_plot(product, data):
    period = [0]
    EQ = [data.loc[product,"Q"]]
    while period[-1] < 12:
        period.append(period[-1] + data.loc[product,"Tp"])
        period.append(period[-1] + data.loc[product,"Td"])
        EQ.append(data.loc[product,"Q"])
        EQ.append(data.loc[product,"Q"])
    # Create inventory list and append values
    
    inventory = [0]
    while len(inventory) < len(period):
        inventory.append(data.loc[product,"Imax"])
        inventory.append(0)
    
    df = {'Month': period, 'Inventory': inventory, "EQ": EQ}

    fig = px.line(df, y = ["Inventory","EQ"] , x = "Month", labels={
                     "variable": "Legend",
                     "value": "Quantity",
                     "index": "Month"
                 })
    return fig 

with st.sidebar:
    data_imput = st.file_uploader("Upload file:")
    if data_imput is None:
        st.error("No file uploaded !!!")
    else:
        raw_data = pd.read_excel(data_imput).set_index("product")
        with st.sidebar.form("Optimizing"):
            options = st.slider("Interest rate (%)", -20,20,0)
            submitted = st.form_submit_button("Optimize")
            if submitted:
                 st.success("Sucessfuly optimized")
                 st.session_state['setup'] = EPQ("setup costs")
                 st.session_state['alt'] = EPQ("alternative costs")
        try:
            csv = convert_df(st.session_state['setup'])
            stiahnut = st.download_button("Download with setup costs", data = csv, file_name="ekon_vyr_davka.csv", mime="text/csv")
            csv = convert_df(st.session_state['alt'])
            stiahnut = st.download_button("Download with alternative costs", data = csv, file_name="alt_ekon_vyr_davka.csv", mime="text/csv")
        except:
            st.error("Press optimize !!!")
        prod = st.multiselect("Choose a product to be analysed", list(raw_data.index.values), max_selections=(1))
        
tab_1, tab_2, tab_3 = st.tabs(["ð¡ Model description","ð Data exploration", "ð Graphical analysis"])
with  tab_1:
    st.markdown("""
            ### Model describtion
            """)
    st.markdown("""VÃ½skum v oblasti modelov zÃ¡sob sa zaÄal klasickÃ½m deterministickÃ½m modelom zÃ¡sob EOQ (Economic order quantity), ktorÃ½ bol prvÃ½ krÃ¡t publikovanÃ½ Fordom W. Harrisom v roku 1913. Tento model stanovuje optimÃ¡lnu veÄ¾kosÅ¥ objednÃ¡vky surovÃ­n alebo komponentov vzhÄ¾adom na nÃ¡klady spojenÃ© s ich skladovanÃ­m a nÃ¡klady spojenÃ© s ich dodanÃ­m. Model EOQ bol publikovanÃ½ s cieÄ¾om poskytnÃºÅ¥ manaÅ¾Ã©rom nÃ¡stroj umoÅ¾ÅujÃºci zjednoduÅ¡enie rozhodovania pri objednÃ¡vanÃ­ surovÃ­n a komponentov od dodÃ¡vateÄ¾ov. NeskÃ´r bol v roku 1918 E.W. Taftom publiokovanÃ½ model EPQ(Economic production quantity) ktorÃ½ je rozÅ¡Ã­renÃ­m modelu EOQ. HlavnÃ½m rozdielom tÃ½chto dvoch modelov je, Å¾e model EOQ predpokladÃ¡ Å¾e objednanÃ© mnoÅ¾stvo je k dispozÃ­cii ihneÄ po objednanÃ­ v celej dÃ¡vke. ZnamenÃ¡ to, Å¾e vÃ½robky sÃº obstarÃ¡vanÃ© z inej spoloÄnosti. Naopak EPQ model predpokladÃ¡ Å¾e spoloÄnosÅ¥ vyrÃ¡ba svoje vlastnÃ© vÃ½robky, ktorÃ© sÃº k dispozÃ­cii pre daÄºsie spracovanie Äi predaj jeden po druhom a nie v dÃ¡vkach. Z tÃ½chto dÃ´vodov je model EPQ vhodnejÅ¡Ã­ na aplikÃ¡ciu vo vÃ½robnÃ½ch podnikoch. Predpoklady tohto modelu sÃº ale znaÄne limitujÃºce.
              JednÃ½m z predpokladov je jednoÃºrovÅovÃ½ produkÄnÃ½ proces. V sÃºÄasnosti sÃº ale produkÄnÃ© procesy natoÄ¾ko zloÅ¾itÃ©, Å¾e dodrÅ¾aÅ¥ tento predpoklad je takmer nemoÅ¾nÃ©. DneÅ¡nÃ© vÃ½robky sÃº Äasto veÄ¾mi sofistikovanÃ© a vyÅ¾adujÃº nÃ¡roÄnÃ© procesy zloÅ¾enÃ© z viacerÃ½ch operÃ¡cii. ÄalÅ¡Ã­m nedostatkom je, Å¾e modely typu EOQ nezahÅÅajÃº kapacitnÃ© ohraniÄenia. V praxi to znamenÃ¡, Å¾e plÃ¡novaÄ musÃ­ kapacitnÃ© ohraniÄenia zohladniÅ¥ ex post, ÄÃ­m sa mÃ´Å¾e odkloniÅ¥ od optimÃ¡lneho rieÅ¡enia. ÄalÅ¡Ã­m nedostatkom je predpoklad stacionarity zÃ¡kaznÃ­ckych potrieb. Model predpokladÃ¡, Å¾e zÃ¡kaznÃ­k dopytuje produkciu pravidelne v rovnykÃ½ch vÃ½Å¡kach.
               AvÅ¡ak aj napriek tÃ½mto obmedzeniam, nemÃ´Å¾eme podceÅovaÅ¥ vplyv tohto modelu na neskorÅ¡Ã­ vÃ½skum v oblasti nÃ¡strojov optimalizÃ¡cie vÃ½robnÃ½ch dÃ¡vok. PrÃ¡ve z dÃ´vodu reÅ¡triktÃ­vnosti tohto  modelu, boli neskÃ´r vyvynutÃ© inÃ© typy modelov, ktorÃ© nÃ¡m umoÅ¾ÅujÃº presnejÅ¡ie opÃ­saÅ¥ realitu. MnohÃ© z tÃ½chto modelov sÃº zaloÅ¾enÃ© na zÃ¡kladnÃ½ch poznatkoch pochÃ¡dzajÃºcich z modelv EOQ Äi EPQ. Model EOQ je taktieÅ¾ dÃ´leÅ¾itou sÃºÄasÅ¥ou histÃ³rie operaÄnÃ©ho vÃ½skumu, nakÄ¾ko predstavuje jednu z prvÃ½ch publikovanÃ½ch aplikÃ¡cii matematickÃ©ho modelu na rozhodovanie v podnikanÃ­.
             PrvÃ½ z modelov, kde boli zohÄ¾adnenÃ© kapacity vÃ½roby bol ELSP model (Economic lot scheduling problem). CieÄ¾om modelu je nÃ¡jsÅ¥ takÃ½ vÃ½robnÃ½ plÃ¡n, ktorÃ½ je prÃ­pustnÃ½ vzhÄ¾adom na kapacitnÃ© ohraniÄenie, priÄom viacerÃ© vÃ½robky sÃº vyrÃ¡banÃ© v dÃ¡vkach na jednom stroji a sÃºÄasne minimalizuje nÃ¡klady. Tento model vÅ¡ak predpokladal iba stacionÃ¡rny dopyt. Napriek reÅ¡triktÃ­vnym predpokladom je ELSP kvÃ´li svojej nelineÃ¡rnosti, kombinatorickÃ½m vlastnostiam a zloÅ¾itosti vÅ¡eobecne znÃ¡my ako NP-Å¥aÅ¾kÃ½ problÃ©m.
                ÄalÅ¡Ã­m modelom ktorÃ½ uÅ¾ predpokladal variabilnÃ½ dopyt v Äase (jednÃ¡ sa o dynamickÃ½ model) bol Wagner-Whitin. Je to model, ktorÃ½ predpokladÃ¡ koneÄnÃ½ plÃ¡novacÃ­ horizont ktorÃ½ je rozdelenÃ½ na niekoÄ¾ko podobdobÃ­. Dopyt je stanevenÃ½ pre kaÅ¾dÃ© podobdobie zvlÃ¡Å¡Å¥ a mÃ´Å¾e sa lÃ­Å¡iÅ¥ v Äase. Wagner-Whitin model vÅ¡ak nezahÅÅa obmedzenie kapacÃ­t.  Je dÃ´leÅ¾itÃ© poznamenaÅ¥ Å¾e napriek tomu Å¾e pomerne sofistikovanÃ© modely zÃ¡sob boli vytvorenÃ© uÅ¾ v 90-tych rokoch 20. soroÄia, stÃ¡le sÃº veÄ¾mi nÃ¡roÄnÃ© na rieÅ¡enie. ZvÃ¤ÄÅ¡a sÃº to Ãºlohy NP-Å¥aÅ¾kÃ©. V prÃ­pade aplikÃ¡cii takÃ½chto modelov na reÃ¡lne produkÄnÃ© procesy sa vyvÃ½jajÃº rÃ´zne heuristiky a meta-heuristiky ktorÃ© poskytujÃº aspoÅ sub-optimÃ¡lne rieÅ¡ienia. V praxi sa totiÅ¾ stretÃ¡vame s Ãºlohami veÄ¾kÃ½ch rozmerov, ktorÃ© by ani sÃºÄasnÃ¡ vÃ½poÄtovÃ¡ technika nedokÃ¡zala rieÅ¡iÅ¥.""")
with  tab_2:
    try: 
        st.dataframe(st.session_state['setup'],use_container_width=True)
        st.dataframe(st.session_state['alt'],use_container_width=True)
    except:
        st.error("Upload a file and press optimize !!!")
with tab_3:
    try:
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Model with respect to setup costs")
            st.plotly_chart(show_product(prod[0], st.session_state['setup'])[0], use_container_width=True, sharing="streamlit", theme="streamlit")
            with st.expander("Data"):
                st.dataframe(show_product(prod[0], st.session_state['setup'])[1],use_container_width=True)
            st.plotly_chart(inv_plot(prod[0], st.session_state['setup']), use_container_width=True, sharing="streamlit", theme="streamlit")
        with col2:
            
            st.write("### Model with respect to alternative costs")
            st.plotly_chart(show_product(prod[0], st.session_state['alt'])[0], use_container_width=True, sharing="streamlit", theme="streamlit")
            with st.expander("Data"):
                st.dataframe(show_product(prod[0], st.session_state['alt'])[1],use_container_width=True)
            st.plotly_chart(inv_plot(prod[0], st.session_state['alt']), use_container_width=True, sharing="streamlit", theme="streamlit")
            
    except:
        st.error("Choose product to be analysed")
        
