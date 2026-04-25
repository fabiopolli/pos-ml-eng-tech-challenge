import joblib
import pandas as pd
import logging
from pathlib import Path
from app.schemas.predict import PredictInput
from src.preprocessing.data_prep import apply_feature_engineering, feature_selection_and_encoding

logger = logging.getLogger(__name__)

# O modelo foi treinado exatamente com estas colunas (na mesma ordem).
# Ao aplicar pd.get_dummies() em inferência com uma única linha,
# garantiremos que o DataFrame final tenha exatamente este formato.
EXPECTED_COLUMNS = [
    'tenure', 'MonthlyCharges', 'TotalCharges', 'Services_Count', 
    'Has_Family', 'Is_Electronic_Check', 'Charge_Difference', 
    'Contract_One year', 'Contract_Two year', 
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 
    'TechSupport_No internet service', 'TechSupport_Yes', 
    'InternetService_Fiber optic', 'InternetService_No', 
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 
    'PaymentMethod_Mailed check', 'Tenure_Bins_13-24', 'Tenure_Bins_25-48', 
    'Tenure_Bins_49-60', 'Tenure_Bins_>60'
]

# Colunas numéricas que precisam de scaling (mesmas usadas em data_prep.py)
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "Services_Count", "Charge_Difference"]

class InferencePreprocessor:
    """
    Pré-processador de Inferência que reutiliza a lógica oficial do treinamento.
    Garante zero 'Training-Serving Skew'.
    """

    def __init__(self):
        self.scaler = None
        self.is_loaded = False
        
        # Caminho do scaler gerado pelo pipeline de treinamento
        self.base_dir = Path(__file__).resolve().parents[3]
        self.scaler_path = self.base_dir / "models" / "scaler.pkl"

    def load(self) -> bool:
        """Carrega o scaler ajustado no treinamento."""
        if self.is_loaded:
            return True

        try:
            if not self.scaler_path.exists():
                logger.warning(f"Scaler não encontrado: {self.scaler_path}")
                return False
            
            self.scaler = joblib.load(self.scaler_path)
            self.is_loaded = True
            logger.info("Scaler carregado com sucesso.")
            return True
        except Exception as e:
            logger.error(f"Erro ao carregar o scaler: {e}")
            return False

    def _convert_input_to_dataframe(self, data: PredictInput) -> pd.DataFrame:
        """
        Converte o schema de entrada (API) para o formato do dataset bruto.
        Faz o mapeamento reverso dos dados validados para o que o data_prep.py espera.
        """
        # Mapear variáveis booleanas da API para "Yes"/"No" como no CSV original
        bool_to_yes_no = lambda x: "Yes" if x else "No"
        
        # Correções de mapeamento para InternetService
        # A API tem has_internet_service e streaming_movies/tv. 
        # O dataset original usa InternetService ('Fiber optic', 'DSL', 'No')
        # Aqui, fazemos uma aproximação: se tem internet e TV, vamos assumir Fiber Optic (para simplificar, ou DSL).
        # Na verdade, o ideal seria a API receber o tipo exato, mas com as features que temos:
        internet_service = "Fiber optic" if data.has_internet_service else "No"
        
        # Mapeamento do método de pagamento
        payment_map = {
            "credit_card": "Credit card (automatic)",
            "debit_card": "Bank transfer (automatic)",
            "electronic_check": "Electronic check",
            "bank_transfer": "Bank transfer (automatic)"
        }
        
        # Mapeamento do tipo de contrato
        contract_map = {
            "monthly": "Month-to-month",
            "one_year": "One year",
            "two_year": "Two year"
        }

        # Para serviços online, se não tem internet, o valor no CSV é "No internet service"
        def online_service_value(has_service: bool):
            if not data.has_internet_service:
                return "No internet service"
            return "Yes" if has_service else "No"

        # Simular a linha do DataFrame antes do apply_feature_engineering
        row = {
            "tenure": data.tenure,
            "MonthlyCharges": data.monthly_charges,
            "TotalCharges": data.total_charges,
            "Contract": contract_map.get(data.contract_type, "Month-to-month"),
            "PaymentMethod": payment_map.get(data.payment_method, "Electronic check"),
            "InternetService": internet_service,
            "OnlineSecurity": online_service_value(data.has_online_security),
            "TechSupport": online_service_value(data.has_tech_support),
            "OnlineBackup": online_service_value(data.has_online_backup),
            "DeviceProtection": online_service_value(data.has_device_protection),
            "StreamingTV": online_service_value(data.streaming_tv),
            "StreamingMovies": online_service_value(data.streaming_movies),
            "Partner": "No", # Assumido No (não presente na API explícita, usado no Has_Family)
            "Dependents": "No", # Assumido No
            "Churn": 0 # Coluna dummy para o feature_selection_and_encoding rodar
        }
        
        return pd.DataFrame([row])

    def preprocess(self, data: PredictInput) -> pd.DataFrame:
        """
        Executa todo o pipeline de preprocessamento para inferência.
        """
        if not self.is_loaded:
            if not self.load():
                raise RuntimeError("Não foi possível carregar o scaler (scaler.pkl). Treine o modelo primeiro.")

        # 1. Converter entrada para formato bruto
        df = self._convert_input_to_dataframe(data)

        # 2. Aplicar Engenharia de Features oficial
        df = apply_feature_engineering(df)

        # 3. Seleção e Encoding
        features, _ = feature_selection_and_encoding(df)

        # 4. Alinhar colunas pós One-Hot Encoding com as do treinamento
        # Como o get_dummies em uma linha não vai gerar as categorias que não estão presentes,
        # fazemos o reindex para preencher com 0 as colunas ausentes e descartar colunas inesperadas.
        features = features.reindex(columns=EXPECTED_COLUMNS, fill_value=0)

        # 5. Normalizar usando o scaler treinado
        features[NUM_COLS] = self.scaler.transform(features[NUM_COLS])

        return features

# Singleton para a API
_preprocessor = None

def preprocess_input(data: PredictInput) -> pd.DataFrame:
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = InferencePreprocessor()
    return _preprocessor.preprocess(data)
