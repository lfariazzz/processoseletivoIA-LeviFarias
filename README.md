# Processo Seletivo – Intensivo Maker | AI

---

👤 **Identificação**

**Nome Completo:** Levi Farias Leite
**GitHub:** [lfariazzz](https://github.com/lfariazzz)

---

### 1️⃣ Resumo da Arquitetura do Modelo

A CNN implementada em `train_model.py` foi projetada com foco em simplicidade e eficiência para Edge AI, priorizando leveza sem sacrificar acurácia.

A rede é composta por dois blocos convolucionais seguidos de um classificador denso. O primeiro bloco aplica 32 filtros 3×3 com ativação ReLU para extrair bordas e texturas simples, seguido de MaxPooling 2×2 para redução espacial. O segundo bloco aplica 64 filtros 3×3 para combinar padrões mais complexos, novamente seguido de MaxPooling. Após o Flatten, uma camada Dense de 64 neurônios com Dropout de 30% realiza a classificação intermediária. A camada de saída possui 10 neurônios com Softmax — uma probabilidade para cada dígito de 0 a 9.

**Total de parâmetros:** 121.930 (476.29 KB) — modelo leve e adequado para Edge AI.

Duas camadas convolucionais são suficientes para o MNIST atingir acurácia acima de 99%, mantendo o modelo compacto para conversão eficiente em TFLite. Arquiteturas mais profundas agregariam custo computacional sem ganho relevante nesse dataset.

---

### 2️⃣ Bibliotecas Utilizadas

| Biblioteca | Versão | Uso |
|---|---|---|
| `tensorflow` | 2.x | Framework principal — treino, conversão TFLite e inferência |
| `numpy` | 1.x | Manipulação de arrays e validação da inferência |

Dependências completas listadas em `requirements.txt`.

---

### 3️⃣ Técnica de Otimização do Modelo

Foi aplicada **Dynamic Range Quantization** no arquivo `optimize_model.py`:

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

Essa técnica converte os pesos do modelo de `float32` para `int8` em tempo de conversão, enquanto as ativações são quantizadas dinamicamente durante a inferência. Não requer dataset de calibração e é compatível com qualquer hardware — CPU ou microcontroladores.

**Trade-off:** quanto mais agressiva a quantização, menor o modelo e mais rápida a inferência, porém maior o risco de degradação de acurácia. Para o MNIST, a Dynamic Range Quantization apresenta perda inferior a 0.2% com redução de ~73% no tamanho — equilíbrio ideal para Edge AI.

---

### 4️⃣ Resultados Obtidos

**Treinamento:**

| Métrica | Valor |
|---|---|
| Acurácia no conjunto de teste | **99.25%** |
| Loss no conjunto de teste | **0.0251** |
| Acurácia de validação (última época) | **99.15%** |

**Otimização:**

| Métrica | Valor |
|---|---|
| Tamanho model.tflite (Dynamic Range Quantization) | **128.3 KB** |
| Acurácia do modelo .tflite (100 amostras) | **100.0%** |

O modelo foi reduzido em ~73% sem perda de acurácia nas amostras testadas.

---

### 5️⃣ Comentários Adicionais (Opcional)

**Decisões técnicas importantes:**
- O Dropout de 30% foi incluído apenas na camada densa. Em inferência o Dropout é desativado automaticamente pelo TFLite, não impactando o desempenho em produção.
- O limite de 5 épocas é mais que suficiente para o MNIST convergir com essa arquitetura, sendo compatível com o ambiente de CI em CPU.
- O `batch_size=64` equilibra velocidade de treino e estabilidade do gradiente em CPU.

**Limitações:**
- O modelo foi treinado exclusivamente para o MNIST. Para datasets mais complexos seria necessária arquitetura mais profunda ou transfer learning.
- A quantização Full Integer não foi aplicada por exigir dataset de calibração, aumentando a complexidade do pipeline de CI.

**Aprendizados:**
O desafio evidencia que Edge AI não é apenas sobre acurácia — é sobre o equilíbrio entre desempenho, tamanho do modelo e viabilidade de execução em hardware restrito. A conversão para TFLite com quantização é o passo crítico que transforma um modelo de laboratório em uma solução real embarcada.