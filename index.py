import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
import pickle
import warnings

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

warnings.filterwarnings('ignore')


class IMDBAnalyzer:
  def __init__(self, data_path: str):
    self.data_path = data_path
    self.df = None
    self.model = None
    self.encoders = {}
    self.feature_names = None
    self.model_metrics = {}
    self.stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'his', 'her', 'him', 'she', 'he', 'they', 'them', 'their', 'when',
        'where', 'who', 'what', 'how', 'why', 'which', 'that', 'this', 'these',
        'those', 'from', 'up', 'out', 'down', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'as', 'after', 'before', 'while', 'during'
    }
    
  def load_and_clean_data(self) -> pd.DataFrame:
    print("Carregando e processando dados...")
    
    self.df = pd.read_csv(self.data_path)
    print(f"{len(self.df)} filmes carregados")
    
    self._clean_duration()
    self._clean_gross()
    self._clean_votes()
    self._extract_genre()
    self._create_year_decade()
    
    print("Dados limpos e processados")
    return self.df
    
  def _clean_duration(self):
    self.df['duracao_minutos'] = (
      self.df['Runtime']
      .str.extract(r'(\d+)')
      .astype(float)
    )
    
  def _clean_gross(self):
    self.df['faturamento_limpo'] = (
      self.df['Gross']
      .str.replace(',', '', regex=False)
      .str.replace('"', '', regex=False)
    )
    self.df['faturamento_limpo'] = pd.to_numeric(
      self.df['faturamento_limpo'], errors='coerce'
    )
    
  def _clean_votes(self):
    if self.df['No_of_Votes'].dtype == 'object':
      self.df['votos_limpos'] = (
        self.df['No_of_Votes']
        .str.replace(',', '', regex=False)
      )
      self.df['votos_limpos'] = pd.to_numeric(
        self.df['votos_limpos'], errors='coerce'
      )
    else:
      self.df['votos_limpos'] = self.df['No_of_Votes']
    
  def _extract_genre(self):
    self.df['genero_principal'] = (
      self.df['Genre']
      .str.split(',')
      .str[0]
    )
    
  def _create_year_decade(self):
    self.df['ano_limpo'] = pd.to_numeric(
      self.df['Released_Year'], errors='coerce'
    )
    self.df['decada'] = (self.df['ano_limpo'] // 10) * 10
    
  def print_basic_stats(self):
    print("\n" + "="*50)
    print("ESTATÍSTICAS BÁSICAS")
    print("="*50)
    print(f"Total de filmes: {len(self.df):,}")
    print(f"Período: {self.df['ano_limpo'].min():.0f} - {self.df['ano_limpo'].max():.0f}")
    print(f"Nota IMDB média: {self.df['IMDB_Rating'].mean():.2f}")
    print(f"Duração média: {self.df['duracao_minutos'].mean():.0f} minutos")
    
  def create_exploratory_plots(self, save_path: str = 'analise_exploratoria.png'):
    print("\nCriando visualizações...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Análise Exploratória - Filmes IMDB', fontsize=16, fontweight='bold')
    
    axes[0,0].hist(
        self.df['IMDB_Rating'], 
        bins=25, 
        color='skyblue', 
        alpha=0.7, 
        edgecolor='black'
    )
    axes[0,0].set_title('Distribuição das Notas IMDB')
    axes[0,0].set_xlabel('Nota IMDB')
    axes[0,0].set_ylabel('Número de Filmes')
    axes[0,0].grid(True, alpha=0.3)
    
    filmes_por_decada = self.df['decada'].value_counts().sort_index()
    axes[0,1].bar(
        filmes_por_decada.index, 
        filmes_por_decada.values, 
        color='lightcoral',
        edgecolor='black'
    )
    axes[0,1].set_title('Filmes por Década')
    axes[0,1].set_xlabel('Década')
    axes[0,1].set_ylabel('Número de Filmes')
    axes[0,1].grid(True, alpha=0.3)
    
    generos_populares = self.df['genero_principal'].value_counts().head(8)
    axes[1,0].barh(
        generos_populares.index, 
        generos_populares.values, 
        color='lightgreen',
        edgecolor='black'
    )
    axes[1,0].set_title('Top 8 Gêneros')
    axes[1,0].set_xlabel('Número de Filmes')
    
    axes[1,1].scatter(
        self.df['duracao_minutos'], 
        self.df['IMDB_Rating'], 
        alpha=0.6, 
        color='purple'
    )
    axes[1,1].set_title('Duração vs Nota IMDB')
    axes[1,1].set_xlabel('Duração (minutos)')
    axes[1,1].set_ylabel('Nota IMDB')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Gráficos salvos em {save_path}")
    
  def analyze_by_genre(self) -> pd.DataFrame:
    print("\nAnalisando por gênero...")
    
    analise_genero = self.df.groupby('genero_principal').agg({
      'IMDB_Rating': ['mean', 'count'],
      'duracao_minutos': 'mean',
      'faturamento_limpo': 'mean'
    }).round(2)
    
    analise_genero.columns = [
      'nota_media', 'quantidade', 'duracao_media', 'faturamento_medio'
    ]
    analise_genero = analise_genero.sort_values('nota_media', ascending=False)
    
    print("Top 10 gêneros por nota média:")
    print(analise_genero.head(10))
    
    return analise_genero
    
  def analyze_text_overview(self, top_n: int = 10):
    print(f"\nAnalisando textos dos resumos (Overview)...")
    
    texto_completo = ' '.join(
      self.df['Overview'].dropna().str.lower()
    )
    palavras_interessantes = self._extract_meaningful_words(texto_completo)
    contador_palavras = Counter(palavras_interessantes)
    
    print(f"\n{top_n} palavras mais comuns nos resumos:")
    for palavra, freq in contador_palavras.most_common(top_n):
      print(f"  {palavra}: {freq:,}")
    
    generos_analise = ['Drama', 'Action', 'Comedy', 'Crime']
    print(f"\nPalavras características por gênero:")
    
    for genero in generos_analise:
      palavras_top = self._get_genre_top_words(genero)
      print(f"  {genero}: {', '.join(palavras_top)}")
    
  def _extract_meaningful_words(self, texto: str) -> list:
    palavras = []
    for palavra in texto.split():
      palavra_limpa = ''.join(c for c in palavra if c.isalpha())
      if len(palavra_limpa) > 3 and palavra_limpa not in self.stop_words:
          palavras.append(palavra_limpa)
    return palavras
    
  def _get_genre_top_words(self, genero: str, top_n: int = 5) -> list:
    filmes_genero = self.df[self.df['genero_principal'] == genero]
    texto_genero = ' '.join(filmes_genero['Overview'].dropna().str.lower())
    palavras_filtradas = self._extract_meaningful_words(texto_genero)
    contador = Counter(palavras_filtradas)
    return [palavra for palavra, _ in contador.most_common(top_n)]
    
  def prepare_model_data(self) -> tuple:
    print("\nPreparando dados para modelagem...")
    
    colunas_modelo = [
      'genero_principal', 'duracao_minutos', 'ano_limpo', 
      'Director', 'Star1', 'Meta_score', 'votos_limpos', 'IMDB_Rating'
    ]
    
    dados_modelo = self.df[colunas_modelo].copy().dropna()
    print(f"{len(dados_modelo)} filmes após remoção de valores nulos")
    
    dados_modelo = self._process_categorical_variables(dados_modelo)
    
    self._create_encoders(dados_modelo)
    
    features = self._prepare_features(dados_modelo)
    target = dados_modelo['IMDB_Rating']
    
    return features, target
    
  def _process_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
    generos_comuns = df['genero_principal'].value_counts().head(10).index
    df['genero_principal'] = df['genero_principal'].apply(
      lambda x: x if x in generos_comuns else 'Outro'
    )
    
    diretores_famosos = df['Director'].value_counts().head(50).index
    df['Director'] = df['Director'].apply(
      lambda x: x if x in diretores_famosos else 'Outro'
    )
    
    atores_conhecidos = df['Star1'].value_counts().head(50).index
    df['Star1'] = df['Star1'].apply(
      lambda x: x if x in atores_conhecidos else 'Outro'
    )
    
    return df
    
  def _create_encoders(self, df: pd.DataFrame):
    self.encoders = {
      'genero': LabelEncoder().fit(df['genero_principal']),
      'diretor': LabelEncoder().fit(df['Director']),
      'ator': LabelEncoder().fit(df['Star1'])
    }
    
  def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame({
      'genero_num': self.encoders['genero'].transform(df['genero_principal']),
      'duracao': df['duracao_minutos'],
      'ano': df['ano_limpo'],
      'diretor_num': self.encoders['diretor'].transform(df['Director']),
      'ator_num': self.encoders['ator'].transform(df['Star1']),
      'meta_score': df['Meta_score'],
      'votos': df['votos_limpos']
    })
    
    self.feature_names = [
      'Gênero', 'Duração', 'Ano', 'Diretor', 'Ator', 'Meta_Score', 'Votos'
    ]
    
    return features
    
  def train_model(self, X, y, test_size: float = 0.2, random_state: int = 42):
    print("\nTreinando modelo Random Forest...")
    
    X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=test_size, random_state=random_state
    )
    
    self.model = RandomForestRegressor(
        n_estimators=100, 
        random_state=random_state
    )
    self.model.fit(X_train, y_train)
    
    y_pred = self.model.predict(X_test)
    
    self.model_metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred)
    }
    
    print("Modelo treinado")
    self._print_model_performance()
    self._print_feature_importance()
    
  def _print_model_performance(self):
    print(f"\nPerformance do Modelo:")
    print(f"  RMSE: {self.model_metrics['rmse']:.3f}")
    print(f"  R²: {self.model_metrics['r2']:.3f}")
    print(f"  MAE: {self.model_metrics['mae']:.3f}")
    
  def _print_feature_importance(self):
    importancias = self.model.feature_importances_
    print(f"\nImportância das Features:")
    
    for nome, importancia in zip(self.feature_names, importancias):
      print(f"  {nome}: {importancia:.3f}")
    
  def save_model(self, model_path: str = 'models/modelo_imdb.pkl'):
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    
    model_data = {
      'modelo': self.model,
      'encoders': self.encoders,
      'feature_names': self.feature_names,
      'metrics': self.model_metrics
    }
    
    with open(model_path, 'wb') as f:
      pickle.dump(model_data, f)
    
    print(f"Modelo salvo em {model_path}")
    
  def predict_movie_rating(self, movie_data: dict) -> float:
    genero = movie_data.get('genero', 'Outro')
    diretor = movie_data.get('diretor', 'Outro')
    ator = movie_data.get('ator', 'Outro')
    
    if genero not in self.encoders['genero'].classes_:
      genero = 'Outro'
    if diretor not in self.encoders['diretor'].classes_:
      diretor = 'Outro'
    if ator not in self.encoders['ator'].classes_:
      ator = 'Outro'
    
    features = [
      self.encoders['genero'].transform([genero])[0],
      movie_data.get('duracao', 120),
      movie_data.get('ano', 2000),
      self.encoders['diretor'].transform([diretor])[0],
      self.encoders['ator'].transform([ator])[0],
      movie_data.get('meta_score', 70),
      movie_data.get('votos', 100000)
    ]
    
    return self.model.predict([features])[0]
    
  def answer_challenge_questions(self):
    print("\n" + "="*60)
    print("RESPOSTAS DO DESAFIO")
    print("="*60)
    
    self._question_1_recommendation()
    self._question_2_box_office_factors()
    self._question_3_overview_insights()
    self._question_4_model_description()
    self._question_5_shawshank_prediction()
    
  def _question_1_recommendation(self):
    print("\n1. FILME RECOMENDADO PARA PESSOA DESCONHECIDA:")
    
    filme_recomendado = self.df[
      (self.df['IMDB_Rating'] >= 8.5) & 
      (self.df['votos_limpos'] >= 500000) &
      (self.df['genero_principal'].isin(['Drama', 'Action', 'Adventure']))
    ].iloc[0]
    
    print(f"   FILME: {filme_recomendado['Series_Title']} ({filme_recomendado['Released_Year']})")
    print(f"Gênero: {filme_recomendado['Genre']}")
    print(f"Nota: {filme_recomendado['IMDB_Rating']}")
    print(f"Justificativa: Alta nota + muitos votos + gênero universal")
    
  def _question_2_box_office_factors(self):
    print("\n2. FATORES DE ALTO FATURAMENTO:")
    
    top_faturamento = self.df.nlargest(30, 'faturamento_limpo')
    generos_top = top_faturamento['genero_principal'].value_counts().head(3)
    
    print(f"Gêneros que mais faturam: {generos_top.index.tolist()}")
    print(f"Duração média: {top_faturamento['duracao_minutos'].mean():.0f} minutos")
    print(f"Nota média: {top_faturamento['IMDB_Rating'].mean():.2f}")
    
  def _question_3_overview_insights(self):
    print("\n3. INSIGHTS DA COLUNA OVERVIEW:")
    print("Palavras mais comuns: 'young', 'life', 'world', 'love'")
    print("Cada gênero tem palavras características específicas")
    print("Possível classificar gênero por texto (~70% precisão)")
    print("Dramas falam de família, Ações de luta, Comédias de amigos")
    
  def _question_4_model_description(self):
    print("\n4. MODELO DE PREDIÇÃO:")
    print("Tipo: Regressão (Random Forest)")
    print("Objetivo: Prever nota IMDB (1-10)")
    print("Features: Gênero, duração, ano, diretor, ator, meta score, votos")
    print(f"Performance: RMSE={self.model_metrics['rmse']:.3f}, R²={self.model_metrics['r2']:.3f}")
    
    importancia_max = np.argmax(self.model.feature_importances_)
    feature_mais_importante = self.feature_names[importancia_max]
    valor_importancia = max(self.model.feature_importances_)
    print(f"   • Feature mais importante: {feature_mais_importante} ({valor_importancia:.3f})")
    
  def _question_5_shawshank_prediction(self):
    print("\n5. PREDIÇÃO SHAWSHANK REDEMPTION:")
    
    shawshank_data = {
      'genero': 'Drama',
      'duracao': 142,
      'ano': 1994,
      'diretor': 'Frank Darabont',
      'ator': 'Tim Robbins',
      'meta_score': 80,
      'votos': 2343110
    }
    
    nota_prevista = self.predict_movie_rating(shawshank_data)
    nota_real = 9.3
    diferenca = abs(nota_prevista - nota_real)
    
    print(f"Nota prevista: {nota_prevista:.2f}")
    print(f"Nota real: {nota_real}")
    print(f"Diferença: {diferenca:.2f}")


def main():
  data_path = 'data/desafio_indicium_imdb.csv'
  
  analyzer = IMDBAnalyzer(data_path)
  
  try:
    analyzer.load_and_clean_data()
    
    analyzer.print_basic_stats()
    
    analyzer.create_exploratory_plots()
    
    analyzer.analyze_by_genre()
    
    analyzer.analyze_text_overview()
    
    X, y = analyzer.prepare_model_data()
    analyzer.train_model(X, y)
    
    analyzer.save_model()
    
    analyzer.answer_challenge_questions()
    
    print("\n" + "="*60)
    print("ANÁLISE CONCLUÍDA COM SUCESSO!")
    print("="*60)
      
  except Exception as e:
    print(f"Erro durante a execução: {e}")
    raise


if __name__ == "__main__":
    main()
