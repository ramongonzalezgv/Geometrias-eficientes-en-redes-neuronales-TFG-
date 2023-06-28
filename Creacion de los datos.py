import numpy as np
import pandas as pd


# LECTURA DE LOS ARCHIVOS QUE CONTIENEN CADA UNO DE LOS INDICES

AJSL = pd.DataFrame(pd.read_excel("C:\\Users\\ramon\\OneDrive\\Desktop\\Universidad\\TFG\\TFG_VScode\\Indices_teleconexion\\AJSL_indice_norm.xlsx",header=None))
AMO = pd.DataFrame(pd.read_excel("C:\\Users\\ramon\\OneDrive\\Desktop\\Universidad\\TFG\\TFG_VScode\\Indices_teleconexion\\AMOi_indice_norm.xlsx",header=None))
AO = pd.DataFrame(pd.read_excel("C:\\Users\\ramon\\OneDrive\\Desktop\\Universidad\\TFG\\TFG_VScode\\Indices_teleconexion\\AOi_indice_norm.xlsx",header=None))
EA = pd.DataFrame(pd.read_excel("C:\\Users\\ramon\\OneDrive\\Desktop\\Universidad\\TFG\\TFG_VScode\\Indices_teleconexion\\EA_indice_norm.xlsx",header=None))
EAWR = pd.DataFrame(pd.read_excel("C:\\Users\\ramon\\OneDrive\\Desktop\\Universidad\\TFG\\TFG_VScode\\Indices_teleconexion\\EAWR_indice_norm.xlsx",header=None))
ENSO = pd.DataFrame(pd.read_excel("C:\\Users\\ramon\\OneDrive\\Desktop\\Universidad\\TFG\\TFG_VScode\\Indices_teleconexion\\ENSOi_indice_norm.xlsx",header=None))
GJSL = pd.DataFrame(pd.read_excel("C:\\Users\\ramon\\OneDrive\\Desktop\\Universidad\\TFG\\TFG_VScode\\Indices_teleconexion\\GJSL_indice_norm.xlsx",header=None))
MO = pd.DataFrame(pd.read_excel("C:\\Users\\ramon\\OneDrive\\Desktop\\Universidad\\TFG\\TFG_VScode\\Indices_teleconexion\\MO_indice_norm.xlsx",header=None))
NAO = pd.DataFrame(pd.read_excel("C:\\Users\\ramon\\OneDrive\\Desktop\\Universidad\\TFG\\TFG_VScode\\Indices_teleconexion\\NAOi_indice_norm.xlsx",header=None))
NHIE = pd.DataFrame(pd.read_excel("C:\\Users\\ramon\\OneDrive\\Desktop\\Universidad\\TFG\\TFG_VScode\\Indices_teleconexion\\NHIE_indice_norm.xlsx",header=None))
PDO = pd.DataFrame(pd.read_excel("C:\\Users\\ramon\\OneDrive\\Desktop\\Universidad\\TFG\\TFG_VScode\\Indices_teleconexion\\PDOi_indice_norm.xlsx",header=None))
QBO = pd.DataFrame(pd.read_excel("C:\\Users\\ramon\\OneDrive\\Desktop\\Universidad\\TFG\\TFG_VScode\\Indices_teleconexion\\QBO_indice_norm.xlsx",header=None))
SAHEL_PI = pd.DataFrame(pd.read_excel("C:\\Users\\ramon\\OneDrive\\Desktop\\Universidad\\TFG\\TFG_VScode\\Indices_teleconexion\\SAHEL_PI_indice_norm.xlsx",header=None))
SCAND = pd.DataFrame(pd.read_excel("C:\\Users\\ramon\\OneDrive\\Desktop\\Universidad\\TFG\\TFG_VScode\\Indices_teleconexion\\SCAND_indice_norm.xlsx",header=None))
SSW = pd.DataFrame(pd.read_excel("C:\\Users\\ramon\\OneDrive\\Desktop\\Universidad\\TFG\\TFG_VScode\\Indices_teleconexion\\SSW_indice_norm.xlsx",header=None))
ULMO = pd.DataFrame(pd.read_excel("C:\\Users\\ramon\\OneDrive\\Desktop\\Universidad\\TFG\\TFG_VScode\\Indices_teleconexion\\ULMO_indice_norm.xlsx",header=None))
WeMO = pd.DataFrame(pd.read_excel("C:\\Users\\ramon\\OneDrive\\Desktop\\Universidad\\TFG\\TFG_VScode\\Indices_teleconexion\\WeMO_indice_norm.xlsx",header=None))


# SE HOMOGENEIZAN LAS FECHAS

index_list = [AJSL,AMO,AO,EA,EAWR,ENSO,GJSL,MO,NAO,NHIE,PDO,QBO,SAHEL_PI,SCAND,SSW,ULMO,WeMO]

for index in index_list:
    index.columns = ['Date','Index']
    index['Date'] = index['Date'].astype('str')
    index['Date'] = index['Date'].apply(lambda x: x.replace('00:00:00',''))
    index['Date'] = index['Date'].apply(lambda x: x.strip())
    index['Date'] = index['Date'].apply(lambda x: x.replace('/','-'))
    index['Date'] = pd.to_datetime(index['Date'], format = '%Y-%m-%d')


# SE JUNTAN TODOS LOS INDICES EN UN SOLO CONJUNTO DE DATOS

x = pd.merge(ENSO,AJSL,on='Date',how='left').merge(AMO,on='Date',how='left').merge(
AO,on='Date',how='left').merge(EA,on='Date',how='left').merge(EAWR,on='Date',how='left').merge(
NHIE,on='Date',how='left').merge(GJSL,on='Date',how='left').merge(MO,on='Date',how='left').merge(
NAO,on='Date',how='left').merge(PDO,on='Date',how='left').merge(QBO,on='Date',how='left').merge(
SAHEL_PI,on='Date',how='left').merge(SCAND,on='Date',how='left').merge(SSW,on='Date',how='left').merge(
ULMO,on='Date',how='left').merge(WeMO,on='Date',how='left');

x.columns = ['Date','ENSO','AJSL','AMO','AO','EA','EAWR','NHIE','GJSL','MO','NAO','PDO','QBO','SAHEL_PI','SCAND','SSW','ULMO','WeMO']

# SE TRUNCAN LAS FECHAS PARA TENER EL MISMO NUMERO DE OBSERVACIONES PARA TODOS LOS INDICES

fecha_inicio = pd.to_datetime('1981-01-01', format = '%Y-%m-%d')
fecha_fin = pd.to_datetime('2017-12-30', format = '%Y-%m-%d')
df_tc = x[x["Date"] <= fecha_fin]
date_time = df_tc["Date"]