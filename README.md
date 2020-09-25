## Permutation importance
A importância da permutação usa modelos de maneira diferente de tudo o que você viu até agora, e muitas pessoas acham isso confuso no início. Portanto, começaremos com um exemplo para torná-lo mais concreto.

Considere os dados com o seguinte formato:

<img src="table.png">

Queremos prever a altura de uma pessoa quando ela completar 20 anos, usando dados disponíveis aos 10 anos.

Nossos dados incluem recursos úteis (altura aos 10 anos), recursos com pouco poder preditivo (meias compradas), bem como alguns outros recursos que não iremos focar nesta explicação.

A importância da permutação é calculada após o ajuste de um modelo. Portanto, não mudaremos o modelo ou quais previsões obteríamos para um determinado valor de altura, contagem de meias, etc.

Em vez disso, faremos a seguinte pergunta: Se eu embaralhar aleatoriamente uma única coluna dos dados de validação, deixando o destino e todas as outras colunas no lugar, como isso afetaria a precisão das previsões nos dados agora embaralhados?

<img src="table_shuffle.png">

Reordenar aleatoriamente uma única coluna deve causar previsões menos precisas, uma vez que os dados resultantes não correspondem mais a nada observado no mundo real. A precisão do modelo é especialmente afetada se embaralharmos uma coluna na qual o modelo dependeu muito para as previsões. Nesse caso, embaralhar a altura aos 10 anos causaria previsões terríveis. Se, em vez disso, embaralhássemos as meias compradas, as previsões resultantes não sofreriam tanto.

Com esse insight, o processo é o seguinte:
<ol>
<li>Obtenha um modelo treinado.</li>
<li>Misture os valores em uma única coluna, faça previsões usando o conjunto de dados resultante. Use essas previsões e os valores reais de destino para calcular o quanto a função de perda sofreu com o embaralhamento. Essa deterioração de desempenho mede a importância da variável que você acabou de embaralhar.</li>
<li>Retorne os dados para a ordem original (desfazendo a ordem aleatória da etapa 2). Agora repita a etapa 2 com a próxima coluna no conjunto de dados, até que você tenha calculado a importância de cada coluna.</li>
</ol>

## Exemplo de codigo

Nosso exemplo usará um modelo que prevê se um time de futebol / futebol americano terá o vencedor do "Homem do Jogo" com base nas estatísticas do time. O prêmio "Homem do Jogo" é concedido ao melhor jogador do jogo. A construção de modelos não é nosso foco atual, então a célula abaixo carrega os dados e constrói um modelo rudimentar.



```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('FIFA 2018 Statistics.csv')
y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary
feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(n_estimators=100,
                                  random_state=0).fit(train_X, train_y)
```

Aqui está como calcular e mostrar importâncias com a biblioteca eli5:


```python
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())
```

<img src="importances.png">



## Interpretando Importâncias de Permutação (PermutationImportance)

Os valores na parte superior são as características mais importantes, e aqueles na parte inferior importam menos.

O primeiro número em cada linha mostra o quanto o desempenho do modelo diminuiu com um embaralhamento aleatório (neste caso, usando "precisão" como métrica de desempenho).

Como a maioria das coisas na ciência de dados, há alguma aleatoriedade na mudança exata de desempenho de uma coluna embaralhada. Medimos a quantidade de aleatoriedade em nosso cálculo de importância de permutação repetindo o processo com embaralhamento múltiplo. O número após ± mede como o desempenho variou de uma remodelação para a próxima.

Você ocasionalmente verá valores negativos para importâncias de permutação. Nesses casos, as previsões sobre os dados embaralhados (ou ruidosos) eram mais precisas do que os dados reais. Isso acontece quando o recurso não importa (deveria ter uma importância próxima a 0), mas a chance aleatória faz com que as previsões nos dados embaralhados sejam mais precisas. Isso é mais comum com conjuntos de dados pequenos, como o deste exemplo, porque há mais espaço para sorte / acaso.

Em nosso exemplo, o recurso mais importante foram os gols marcados. Isso parece sensato. Os fãs de futebol podem ter alguma intuição sobre se a ordenação de outras variáveis ​​é surpreendente ou não.
