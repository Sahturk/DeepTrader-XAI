import logging
from stable_baselines3 import PPO

class DeepTraderAI:
    def __init__(self):
        self.ta = TechnicalAnalysis()
        self.trader = TradingEngine()
        self.risk_manager = RiskManager()
        self.social_media_analyzer = SocialMediaAnalyzer()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.price_predictor = PricePredictor()
        self.advanced_risk_manager = AdvancedRiskManager()
        self.parameter_optimizer = ParameterOptimizer()
        self.performance_monitor = PerformanceMonitor()
        self.strategy_params = {
            "rsi_buy": 30,
            "rsi_sell": 70,
            "risk_percent": 0.02
        }
        self.model = PPO("MlpPolicy", "CartPole-v1", verbose=1)
    
    def optimize_parameters(self, historical_data):
        try:
            # Basit optimizasyon örneği
            win_rate = len([x for x in historical_data if x['profit'] > 0]) / len(historical_data)
            self.strategy_params["risk_percent"] = min(0.05, max(0.01, win_rate * 0.03))
        except Exception as e:
            logging.error(f"Error optimizing parameters: {e}")
        
    def run_strategy(self, symbol='BTC/USDT'):
        df = DataFeed.get_ohlcv(symbol)
        if df.empty:
            return
        
        df = self.ta.calculate_ta(df)
        patterns = self.ta.detect_patterns(df)
        
        last_close = df['close'].iloc[-1]
        stop_loss = self.risk_manager.dynamic_stop_loss(df)
        amount = self.risk_manager.calculate_position_size(
            self.strategy_params["risk_percent"],
            last_close,
            stop_loss
        )
        
        # Alım/Satım Mantığı
        if df['rsi'].iloc[-1] < self.strategy_params["rsi_buy"] and 'cup_handle' in patterns:
            self.trader.execute_trade(symbol, 'buy', amount, stop_loss)
        elif df['rsi'].iloc[-1] > self.strategy_params["rsi_sell"] or 'head_shoulders' in patterns:
            self.trader.execute_trade(symbol, 'sell', amount)

    def train_reinforcement_model(self):
        try:
            # Takviye öğrenme modelinin eğitilmesi
            self.model.learn(total_timesteps=10000)
        except Exception as e:
            logging.error(f"Error training reinforcement model: {e}")
    
    def analyze_social_media(self, query):
        tweets = self.social_media_analyzer.fetch_tweets(query)
        sentiments = self.social_media_analyzer.analyze_sentiment(tweets)
        return sentiments
    
    def optimize_portfolio(self, returns):
        optimizer = PortfolioOptimizer(returns)
        weights = optimizer.optimize()
        performance = optimizer.calculate_portfolio_performance(weights)
        return weights, performance
    
    def predict_prices(self, data):
        return self.price_predictor.predict(data)
    
    def manage_advanced_risk(self, returns):
        var = self.advanced_risk_manager.calculate_var(returns)
        cvar = self.advanced_risk_manager.calculate_cvar(returns)
        return var, cvar
    
    def optimize_parameters(self, X, y):
        return self.parameter_optimizer.optimize(X, y)
    
    def monitor_performance(self, date, portfolio_value):
        self.performance_monitor.log_performance(date, portfolio_value)
        self.performance_monitor.generate_report()