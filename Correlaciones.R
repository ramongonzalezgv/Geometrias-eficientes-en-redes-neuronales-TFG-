setwd("C:/Users/ramon/OneDrive/Desktop/Universidad/TFG/TFG_VScode/Datos_teleconexion_vs")
df = read.csv('df_tf_nodate.csv',header=T)
attach(df)


# Calcular las correlaciones
cor_matrix <- cor(df,method = 'spearman')

# Crear una matriz vacía para almacenar los valores p de las pruebas de significancia
p_values <- matrix(NA, nrow = ncol(cor_matrix), ncol = ncol(cor_matrix))

# Realizar las pruebas de significancia para cada correlación
for (i in 1:(ncol(cor_matrix) - 1)) {
  for (j in (i + 1):ncol(cor_matrix)) {
    result <- cor.test(df[, i], df[, j])
    p_values[i, j] <- result$p.value
    p_values[j, i] <- result$p.value
  }
}

# Aplicar umbral de significancia de 0.05
significant_p_values <- p_values < 0.05

# Imprimir la matriz de correlaciones y los valores p
print(cor_matrix)
print(p_values)

result <- ks.test(AO, "pnorm", mean(AO), sd(AO))

# Imprimir los resultados
print(result)


# Realizar la prueba de independencia
result <- chisq.test(table(AO, WeMO))

# Imprimir los resultados
print(result)

#         indep  Normal   corr 
#ENSO      no      no      si
#AJSL      no      no      si
#AMO       no      no      si
#EA        si      no      si
#EAWR      si      no      si
#NHIE      si      no      si
#GJSL      no      no      si
#MO        si      no      si   
#NAO       no      no      si
#PDO       no      no      si
#QBO       no      no      no
#SAHEL_PI  si      no      si 
#SCAND     no      no      si
#SSW       no      no      si  
#ULMO      si      no      si  
#WeMO      si      no      si  


xx = cor.test(AO,WeMO,method = 'spearman'); xx$p.value