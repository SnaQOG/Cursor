import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch
from src.analysis.ai_analyzer import (
    AIAnalyzer,
    AIAnalysisResult,
    MarketMetrics,
    ModelConnectionError,
    ModelResponseError
)

class TestAIAnalyzer:
    @pytest.fixture
    def config(self):
        return {
            'AI_ANALYSIS_WEIGHT': 0.3,
            'AI_MIN_CONFIDENCE': 0.7,
            'DEVELOPMENT_MODE': False
        }
        
    @pytest.fixture
    def sample_market_data(self):
        np.random.seed(42)
        return {
            '1h': {
                'close': np.random.random(100) * 100,
                'high': np.random.random(100) * 110,
                'low': np.random.random(100) * 90,
                'volume': np.random.random(100) * 1000
            },
            '4h': {
                'close': np.random.random(25) * 100,
                'high': np.random.random(25) * 110,
                'low': np.random.random(25) * 90,
                'volume': np.random.random(25) * 1000
            }
        }
        
    @pytest.fixture
    def mock_gpt_response(self):
        return {
            'choices': [{
                'message': {
                    'content': """
                    Analysis:
                    1. Trading direction: buy
                    2. Confidence: 0.85
                    3. Key reasons:
                       - Strong upward momentum
                       - Increasing volume
                       - Bullish pattern formation
                    4. Risk assessment:
                       - Moderate volatility
                       - Good risk/reward ratio
                    """
                }
            }]
        }
        
    @pytest.fixture
    def mock_gemini_response(self):
        return Mock(text="""
        Analysis:
        1. Trading direction: buy
        2. Confidence: 0.80
        3. Key reasons:
           - Technical breakout
           - Volume confirmation
           - Positive momentum
        4. Risk assessment:
           - Acceptable volatility
           - Clear support levels
        """)
        
    @pytest.mark.asyncio
    async def test_initialization(self, config):
        """Test Initialisierung des AI Analyzers."""
        analyzer = AIAnalyzer(config)
        
        assert analyzer.enabled == (config['AI_ANALYSIS_WEIGHT'] > 0)
        assert len(analyzer.models) == 0  # Noch keine Modelle initialisiert
        
        # Test SSL Context
        assert analyzer.ssl_context is not None
        if config['DEVELOPMENT_MODE']:
            assert not analyzer.ssl_context.check_hostname
        else:
            assert analyzer.ssl_context.check_hostname
            
    @pytest.mark.asyncio
    async def test_prepare_market_metrics(self, config, sample_market_data):
        """Test Aufbereitung der Marktdaten."""
        analyzer = AIAnalyzer(config)
        metrics = analyzer._prepare_market_metrics(sample_market_data)
        
        assert len(metrics) == len(sample_market_data)
        for timeframe, data in metrics.items():
            assert isinstance(data, MarketMetrics)
            assert data.current_price > 0
            assert isinstance(data.volume_trend, str)
            assert data.volume_ratio > 0
            
    def test_create_analysis_prompt(self, config):
        """Test Erstellung des Analyse-Prompts."""
        analyzer = AIAnalyzer(config)
        metrics = {
            '1h': MarketMetrics(
                current_price=100.0,
                price_change_1h=1.5,
                price_change_24h=5.0,
                volatility=15.0,
                volume_trend='increasing',
                volume_ratio=1.2
            )
        }
        
        prompt = analyzer._create_analysis_prompt(metrics)
        
        assert 'Trading direction' in prompt
        assert 'Confidence score' in prompt
        assert 'Key reasons' in prompt
        assert 'Risk assessment' in prompt
        
    @pytest.mark.asyncio
    async def test_gpt4_analysis(self, config, mock_gpt_response):
        """Test GPT-4 Analyse."""
        with patch('openai.AsyncClient') as mock_client:
            mock_client.return_value.chat.completions.create.return_value = mock_gpt_response
            
            analyzer = AIAnalyzer(config)
            await analyzer.initialize()
            
            result = await analyzer.models[0].analyze("Test prompt")
            
            assert isinstance(result, AIAnalysisResult)
            assert result.direction == 'buy'
            assert result.confidence == 0.85
            assert result.model_name == 'gpt4'
            
    @pytest.mark.asyncio
    async def test_gemini_analysis(self, config, mock_gemini_response):
        """Test Gemini Analyse."""
        with patch('google.generativeai.GenerativeModel') as mock_model:
            mock_model.return_value.generate_content_async.return_value = mock_gemini_response
            
            analyzer = AIAnalyzer(config)
            await analyzer.initialize()
            
            result = await analyzer.models[1].analyze("Test prompt")
            
            assert isinstance(result, AIAnalysisResult)
            assert result.direction == 'buy'
            assert result.confidence == 0.80
            assert result.model_name == 'gemini'
            
    @pytest.mark.asyncio
    async def test_error_handling(self, config):
        """Test Fehlerbehandlung."""
        analyzer = AIAnalyzer(config)
        
        # Test Model Connection Error
        with pytest.raises(ModelConnectionError):
            await analyzer.models[0].analyze("Test prompt")
            
        # Test Model Response Error
        with patch('openai.AsyncClient') as mock_client:
            mock_client.return_value.chat.completions.create.return_value = {
                'choices': [{
                    'message': {
                        'content': "Invalid response format"
                    }
                }]
            }
            
            with pytest.raises(ModelResponseError):
                await analyzer.models[0].analyze("Test prompt")
                
    @pytest.mark.asyncio
    async def test_combine_analyses(self, config):
        """Test Kombination von Analysen."""
        analyzer = AIAnalyzer(config)
        
        analyses = [
            AIAnalysisResult(
                direction='buy',
                confidence=0.85,
                analysis="Strong buy signal",
                model_name='gpt4',
                timestamp=datetime.now()
            ),
            AIAnalysisResult(
                direction='buy',
                confidence=0.75,
                analysis="Moderate buy signal",
                model_name='gemini',
                timestamp=datetime.now()
            )
        ]
        
        combined = analyzer._combine_analyses(analyses)
        
        assert combined['direction'] == 'buy'
        assert combined['confidence'] > 0.7
        assert len(combined['analyses']) == 2
        
    @pytest.mark.asyncio
    async def test_full_analysis_pipeline(self, config, sample_market_data,
                                        mock_gpt_response, mock_gemini_response):
        """Test kompletter Analyse-Pipeline."""
        with patch('openai.AsyncClient') as mock_openai:
            with patch('google.generativeai.GenerativeModel') as mock_gemini:
                mock_openai.return_value.chat.completions.create.return_value = mock_gpt_response
                mock_gemini.return_value.generate_content_async.return_value = mock_gemini_response
                
                analyzer = AIAnalyzer(config)
                await analyzer.initialize()
                
                result = await analyzer.analyze_market(sample_market_data)
                
                assert result is not None
                assert 'direction' in result
                assert 'confidence' in result
                assert 'analyses' in result
                assert result['confidence'] >= config['AI_MIN_CONFIDENCE']
                
    def test_edge_cases(self, config):
        """Test Verhalten bei Grenzfällen."""
        # Test mit deaktivierter AI
        config['AI_ANALYSIS_WEIGHT'] = 0
        analyzer = AIAnalyzer(config)
        assert not analyzer.enabled
        
        # Test mit leeren Marktdaten
        empty_data = {}
        metrics = analyzer._prepare_market_metrics(empty_data)
        assert len(metrics) == 0
        
        # Test mit ungültigen Marktdaten
        invalid_data = {'1h': {'close': []}}
        metrics = analyzer._prepare_market_metrics(invalid_data)
        assert len(metrics) == 0
        
    @pytest.mark.asyncio
    async def test_performance(self, config, sample_market_data):
        """Test Performance bei großen Datensätzen."""
        analyzer = AIAnalyzer(config)
        
        # Generiere große Datenmenge
        large_data = {}
        for i in range(10):  # 10 Timeframes
            large_data[f'{i}h'] = {
                'close': np.random.random(1000) * 100,
                'high': np.random.random(1000) * 110,
                'low': np.random.random(1000) * 90,
                'volume': np.random.random(1000) * 1000
            }
            
        import time
        
        # Zeitmessung für Datenaufbereitung
        start_time = time.time()
        metrics = analyzer._prepare_market_metrics(large_data)
        prep_time = time.time() - start_time
        
        assert prep_time < 1.0  # Sollte unter 1 Sekunde sein 