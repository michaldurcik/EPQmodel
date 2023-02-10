# Import required libraries
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="EPQ",
    page_icon="⚙️",
    layout="wide",
)

st.sidebar.header("EPQ model")

st.markdown("""
            ### ⚙️ Tool for identifying optimal lotsize.
            👨‍💻 Michal Ďurčík\\
            📆 8.2.2023
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
                     "value": "Costs (€)",
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
        
tab_1, tab_2, tab_3 = st.tabs(["💡 Model description","📄 Data exploration", "📈 Graphical analysis"])
with  tab_1:
    st.write("Describe the model")
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
        
