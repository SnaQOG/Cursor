import os
import json
import asyncio
import aiohttp
import ssl
import logging
from typing import Dict, Optional, List, Union
from datetime import datetime
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential

# Custom exceptions
class AIAnalysisError(Exception):
    """Base exception for AI analysis errors."""
    pass

class ModelConnectionError(AIAnalysisError):
    """Raised when connection to AI model fails."""
    pass

class ModelResponseError(AIAnalysisError):
    """Raised when AI model response is invalid."""
    pass

@dataclass
class MarketMetrics:
    """Data class for market metrics."""
    current_price: float
    price_change_1h: float
    price_change_24h: float
    volatility: float
    volume_trend: str
    volume_ratio: float
    
@dataclass
class AIAnalysisResult:
    """Data class for AI analysis results."""
    direction: str
    confidence: float
    analysis: str
    model_name: str
    timestamp: datetime

class BaseAIModel(ABC):
    """Abstract base class for AI models."""
    
    @abstractmethod
    async def analyze(self, prompt: str) -> AIAnalysisResult:
        """Analyze market data and return results."""
        pass
        
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the model."""
        pass

class GPT4Model(BaseAIModel):
    """GPT-4 model implementation."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=tenacity.retry_if_exception_type(ModelConnectionError)
    )
    async def analyze(self, prompt: str) -> AIAnalysisResult:
        """Analyze market data using GPT-4."""
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a professional crypto trader analyzing market data."},
                    {"role": "user", "content": prompt}
                ]
            )
            return self._parse_response(response.choices[0].message.content)
        except Exception as e:
            raise ModelConnectionError(f"GPT-4 analysis failed: {str(e)}")
            
    async def initialize(self) -> None:
        """Initialize OpenAI client."""
        import openai
        self.client = openai.AsyncClient(api_key=self.api_key)
        
    def _parse_response(self, content: str) -> AIAnalysisResult:
        """Parse GPT-4 response into structured format."""
        try:
            direction = 'neutral'
            if 'buy' in content.lower():
                direction = 'buy'
            elif 'sell' in content.lower():
                direction = 'sell'
                
            confidence = 0.0
            for line in content.split('\n'):
                if 'confidence' in line.lower():
                    try:
                        confidence = float(line.split(':')[1].strip().split()[0])
                        break
                    except:
                        pass
                        
            return AIAnalysisResult(
                direction=direction,
                confidence=confidence,
                analysis=content,
                model_name='gpt4',
                timestamp=datetime.now()
            )
        except Exception as e:
            raise ModelResponseError(f"Failed to parse GPT-4 response: {str(e)}")

class GeminiModel(BaseAIModel):
    """Gemini model implementation."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.model = None
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=tenacity.retry_if_exception_type(ModelConnectionError)
    )
    async def analyze(self, prompt: str) -> AIAnalysisResult:
        """Analyze market data using Gemini."""
        try:
            response = await self.model.generate_content_async(prompt)
            return self._parse_response(response.text)
        except Exception as e:
            raise ModelConnectionError(f"Gemini analysis failed: {str(e)}")
            
    async def initialize(self) -> None:
        """Initialize Gemini model."""
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
    def _parse_response(self, content: str) -> AIAnalysisResult:
        """Parse Gemini response into structured format."""
        try:
            direction = 'neutral'
            if 'buy' in content.lower():
                direction = 'buy'
            elif 'sell' in content.lower():
                direction = 'sell'
                
            confidence = 0.0
            for line in content.split('\n'):
                if 'confidence' in line.lower():
                    try:
                        confidence = float(line.split(':')[1].strip().split()[0])
                        break
                    except:
                        pass
                        
            return AIAnalysisResult(
                direction=direction,
                confidence=confidence,
                analysis=content,
                model_name='gemini',
                timestamp=datetime.now()
            )
        except Exception as e:
            raise ModelResponseError(f"Failed to parse Gemini response: {str(e)}")

class AIAnalyzer:
    """Main AI analysis class with improved error handling and modularity."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.enabled = config.get('AI_ANALYSIS_WEIGHT', 0) > 0
        self.models: List[BaseAIModel] = []
        self.logger = logging.getLogger(__name__)
        
        # SSL context for API calls
        self.ssl_context = self._create_ssl_context()
        self.session = None
        
    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context with proper security settings."""
        context = ssl.create_default_context()
        if self.config.get('DEVELOPMENT_MODE', False):
            self.logger.warning("Running in development mode with reduced SSL security")
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        return context
        
    async def initialize(self) -> None:
        """Initialize AI models and session."""
        if not self.enabled:
            self.logger.info("AI analysis is disabled")
            return
            
        # Initialize OpenAI if configured
        if openai_key := os.getenv('OPENAI_API_KEY'):
            model = GPT4Model(openai_key)
            await model.initialize()
            self.models.append(model)
        else:
            self.logger.warning("OpenAI API key not found, GPT analysis will be skipped")
            
        # Initialize Gemini if configured
        if gemini_key := os.getenv('GEMINI_API_KEY'):
            model = GeminiModel(gemini_key)
            await model.initialize()
            self.models.append(model)
        else:
            self.logger.warning("Gemini API key not found, Gemini analysis will be skipped")
            
        # Initialize HTTP session
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=self.ssl_context)
        )
        
    async def close(self) -> None:
        """Clean up resources."""
        if self.session:
            await self.session.close()
            
    def _prepare_market_metrics(self, market_data: Dict) -> Dict[str, MarketMetrics]:
        """Prepare market data for AI analysis."""
        metrics = {}
        for timeframe, data in market_data.items():
            close_prices = np.array(data['close'])
            pct_changes = np.diff(close_prices) / close_prices[:-1] * 100
            volume = np.array(data['volume'])
            avg_volume = np.mean(volume)
            
            metrics[timeframe] = MarketMetrics(
                current_price=data['close'][-1],
                price_change_1h=pct_changes[-1] if len(pct_changes) > 0 else 0,
                price_change_24h=np.sum(pct_changes[-24:]) if len(pct_changes) >= 24 else np.sum(pct_changes),
                volatility=np.std(pct_changes) * np.sqrt(252),
                volume_trend='increasing' if volume[-1] > avg_volume else 'decreasing',
                volume_ratio=volume[-1] / avg_volume
            )
        return metrics
        
    def _create_analysis_prompt(self, metrics: Dict[str, MarketMetrics]) -> str:
        """Create detailed prompt for AI analysis."""
        prompt = "Analyze the following market data and provide trading signals:\n\n"
        
        for timeframe, data in metrics.items():
            prompt += f"\n{timeframe} Timeframe:\n"
            prompt += f"- Current Price: {data.current_price:.8f}\n"
            prompt += f"- 1h Price Change: {data.price_change_1h:.2f}%\n"
            prompt += f"- 24h Price Change: {data.price_change_24h:.2f}%\n"
            prompt += f"- Volatility: {data.volatility:.2f}%\n"
            prompt += f"- Volume Trend: {data.volume_trend}\n"
            prompt += f"- Volume Ratio: {data.volume_ratio:.2f}\n"
            
        prompt += "\nProvide the following analysis:\n"
        prompt += "1. Trading direction (buy/sell/neutral)\n"
        prompt += "2. Confidence score (0-1)\n"
        prompt += "3. Key reasons for the recommendation\n"
        prompt += "4. Risk assessment\n"
        
        return prompt
        
    async def analyze_market(self, market_data: Dict) -> Optional[Dict]:
        """Analyze market data using available AI models."""
        if not self.enabled or not self.models:
            return None
            
        try:
            # Prepare market data
            metrics = self._prepare_market_metrics(market_data)
            prompt = self._create_analysis_prompt(metrics)
            
            # Collect analyses from all models
            analyses = []
            tasks = [model.analyze(prompt) for model in self.models]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Model analysis failed: {str(result)}")
                    continue
                analyses.append(result)
                
            if not analyses:
                return None
                
            # Combine analyses
            combined = self._combine_analyses(analyses)
            
            if combined and combined['confidence'] >= self.config['AI_MIN_CONFIDENCE']:
                return combined
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error in market analysis: {str(e)}")
            return None
            
    def _combine_analyses(self, analyses: List[AIAnalysisResult]) -> Optional[Dict]:
        """Combine analyses from multiple models with weighted voting."""
        if not analyses:
            return None
            
        # Weight votes by confidence
        direction_votes = {'buy': 0.0, 'sell': 0.0, 'neutral': 0.0}
        total_confidence = 0.0
        
        for analysis in analyses:
            direction_votes[analysis.direction] += analysis.confidence
            total_confidence += analysis.confidence
            
        if total_confidence == 0:
            return None
            
        # Normalize votes
        for direction in direction_votes:
            direction_votes[direction] /= total_confidence
            
        # Select direction with highest weighted vote
        final_direction = max(direction_votes.items(), key=lambda x: x[1])[0]
        
        # Calculate combined confidence
        combined_confidence = direction_votes[final_direction]
        
        return {
            'direction': final_direction,
            'confidence': combined_confidence,
            'analyses': [
                {
                    'model': a.model_name,
                    'direction': a.direction,
                    'confidence': a.confidence,
                    'analysis': a.analysis,
                    'timestamp': a.timestamp.isoformat()
                }
                for a in analyses
            ]
        } 