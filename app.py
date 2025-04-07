
import urllib.parse
import numpy as np
import re
import random
import streamlit as st
from datetime import datetime
import requests
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from nltk import ngrams
import json
import logging
from collections import defaultdict
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import pickle
import os
from newsapi import NewsApiClient
from dotenv import load_dotenv
from functools import lru_cache


load_dotenv()

try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    print("NLTK resources could not be downloaded. Using basic tokenization.")

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# language mapping 
language_mapping = {
    'English': 'en',
    'Spanish': 'es',
    'French': 'fr',
    'German': 'de',
    'Italian': 'it',
    'Portuguese': 'pt',
    'Russian': 'ru',
    'Chinese': 'zh',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Arabic': 'ar',
    'Hindi': 'hi',
    'Bengali': 'bn',
    'Turkish': 'tr',
    'Vietnamese': 'vi',
    'Thai': 'th',
    'Indonesian': 'id',
    'Malay': 'ms',
    'Swedish': 'sv',
    'Norwegian': 'no',
    'Danish': 'da',
    'Finnish': 'fi',
    'Czech': 'cs',
    'Hungarian': 'hu',
    'Polish': 'pl',
    'Romanian': 'ro',
    'Bulgarian': 'bg',
    'Greek': 'el',
    'Ukrainian': 'uk',
    'Hebrew': 'he',
    'Thai': 'th',
    'Filipino': 'tl',
    'Swahili': 'sw',
    'Malayalam': 'ml',
    'Tamil': 'ta',
    'Telugu': 'te',
}

# Set up logging
logging.basicConfig(filename='bot.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

with open('intents.json') as f:
    intents = json.load(f)

intents_list = list(intents.keys())

class VocabularyManager:
    def __init__(self, vocab_path='vocabulary.pkl'):
        self.vocab_path = vocab_path
        self.vocabulary = {'unigrams': [], 'bigrams': []}
        self.timestamp = None
    
    def save_vocabulary(self, unigrams, bigrams):
        """Save vocabulary with timestamp"""
        self.vocabulary = {
            'unigrams': sorted(list(set(unigrams))),
            'bigrams': sorted(list(set(bigrams))),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.vocab_path, 'wb') as f:
            pickle.dump(self.vocabulary, f)
            
    def load_vocabulary(self):
        """Load existing vocabulary"""
        try:
            if os.path.exists(self.vocab_path):
                with open(self.vocab_path, 'rb') as f:
                    self.vocabulary = pickle.load(f)
                return True
            return False
        except Exception as e:
            logging.error(f"Error loading vocabulary: {e}")
            return False
            
    def get_vocabulary(self):
        """Get current vocabulary"""
        return self.vocabulary['unigrams'], self.vocabulary['bigrams']

class NeuralNetwork:
    """Neural Network for intent classification with improved architecture"""
    def __init__(self, layer_sizes=None, input_size=None):
        if layer_sizes is None:
            # layer_sizes = [100, 64, 32, len(intents_list)]
            layer_sizes = [input_size, 64, 32, len(intents_list)]
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        self.initialize_parameters()

    def initialize_parameters(self):
        """Initialize weights and biases with He initialization"""
        for i in range(len(self.layer_sizes)-1):
            limit = np.sqrt(2 / self.layer_sizes[i])
            self.weights.append(np.random.randn(self.layer_sizes[i+1], self.layer_sizes[i]) * limit)
            self.biases.append(np.random.randn(self.layer_sizes[i+1], 1))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / exp_z.sum(axis=0, keepdims=True)

    def forward(self, x):
        """Forward pass with cache for backpropagation"""
        activations = [x]
        zs = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(w, activations[-1]) + b
            zs.append(z)
            if i == len(self.weights)-1:  # output layer
                activations.append(self.softmax(z))
            else:  # hidden layers
                activations.append(self.relu(z))
        return activations, zs

    def compute_loss(self, output, target):
        """Cross-entropy loss"""
        epsilon = 1e-12
        output = np.clip(output, epsilon, 1. - epsilon)
        return -np.sum(target * np.log(output))

    def backward(self, activations, zs, target):
        """Backpropagation with L2 regularization"""
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]

        # Output layer error
        delta = (activations[-1] - target)

        # Backpropagate through layers
        for l in range(len(self.layer_sizes)-2, -1, -1):
            grad_w[l] = np.dot(delta, activations[l].T)
            grad_b[l] = delta

            if l > 0:
                delta = np.dot(self.weights[l].T, delta) * self.relu_derivative(zs[l-1])

        return grad_w, grad_b

    def update_parameters(self, grad_w, grad_b, learning_rate):
        """Update parameters with momentum"""
        if not hasattr(self, 'momentum_w'):
            self.momentum_w = [np.zeros(w.shape) for w in self.weights]
            self.momentum_b = [np.zeros(b.shape) for b in self.biases]

        beta = 0.9  # momentum coefficient
        for i in range(len(self.weights)):
            self.momentum_w[i] = beta * self.momentum_w[i] + (1-beta) * grad_w[i]
            self.momentum_b[i] = beta * self.momentum_b[i] + (1-beta) * grad_b[i]

            self.weights[i] -= learning_rate * self.momentum_w[i]
            self.biases[i] -= learning_rate * self.momentum_b[i]

    def save_model(self, filename):
        """Save model to file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'weights': self.weights,
                'biases': self.biases,
                'layer_sizes': self.layer_sizes
            }, f)

    def load_model(self, filename):
        """Load model from file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.weights = data['weights']
            self.biases = data['biases']
            self.layer_sizes = data['layer_sizes']

class ChatbotProcessor:
    """Handles text processing and feature extraction"""
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vocabulary = []
        self.bigram_vocab = []
        self.entity_patterns = {
            'date': r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b',
            'time': r'\b(\d{1,2}:\d{2}\s*(?:AM|PM)?)\b',
            'location': r'\b(in|at|near|around)\s+([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\b',
            'number': r'\b(\d+)\b'
        }
        self.abbreviations = {
            "what's": "what is",
            "i'm": "i am",
            "don't": "do not",
            "can't": "cannot",
            "won't": "will not",
            "you're": "you are"
        }

    def preprocess_text(self, text):
        """Tokenize, stem, and remove stopwords"""
        # Convert to lowercase
        text = text.lower()
        
        # Replace abbreviations
        for abbrev, full in self.abbreviations.items():
            text = text.replace(abbrev, full)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-z0-9\s\?\.\!,]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) 
                 for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return tokens

    def build_vocabulary(self, patterns):
        """Build vocabulary from training patterns using bi-grams"""
        vocab = set()
        bigram_vocab = set()
        
        for pattern in patterns:
            tokens = self.preprocess_text(pattern)
            vocab.update(tokens)
            
            # Generate bigrams
            bigrams = ['_'.join(bigram) for bigram in ngrams(tokens, 2)]
            bigram_vocab.update(bigrams)
        
        self.vocabulary = sorted(list(vocab))
        self.bigram_vocab = sorted(list(bigram_vocab))

    def text_to_vector(self, text):
        """Convert text to feature vector using n-grams and TF-IDF"""
        tokens = self.preprocess_text(text)
        
        # Unigram features
        unigram_vector = np.zeros(len(self.vocabulary))
        for token in tokens:
            if token in self.vocabulary:
                unigram_vector[self.vocabulary.index(token)] += 1
        
        # Bigram features
        bigrams = ['_'.join(bigram) for bigram in ngrams(tokens, 2)]
        bigram_vector = np.zeros(len(self.bigram_vocab))
        for bigram in bigrams:
            if bigram in self.bigram_vocab:
                bigram_vector[self.bigram_vocab.index(bigram)] += 1
        
        # Combine features
        combined_vector = np.concatenate([unigram_vector, bigram_vector])
        return combined_vector.reshape(-1, 1)

    def extract_entities(self, text):
        """Extract entities from text using regex patterns"""
        entities = {}
        text_lower = text.lower()

        # Handle weather-specific patterns
        if 'weather' in text_lower:
            # Remove question words and get the location
            clean_text = re.sub(r'what\'?s|what is|how\'?s|tell me|forecast|report|temperature', '', text_lower, flags=re.IGNORECASE)
            clean_text = clean_text.replace('weather', '').strip()
            
            # Extract potential location (everything except stopwords)
            tokens = [token for token in word_tokenize(clean_text) 
                    if token not in self.stop_words and len(token) > 2]
            
            if tokens:
                location = ' '.join(tokens)
                entities['location'] = [location.title()]
        

        # Extract Location using prepositions
        location_keywords = ['in', 'at', 'near', 'around', 'for', 'of']
        words = word_tokenize(text)

        for i, word in enumerate(words):
            if word.lower() in location_keywords and i+1 < len(words):
                location = ' '.join(words[i+1:])  # Take all words after the preposition
                if 'location' not in entities:
                    entities['location'] = []
                entities['location'].append(location)
                break

        for entity_type, pattern in self.entity_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if entity_type not in entities:
                    entities[entity_type] = []
                entities[entity_type].append(match.group())
        return entities

    @lru_cache(maxsize=1000)
    def _preprocess_text_cached(self, text):
        """Cached version of text preprocessing"""
        return self.preprocess_text(text)

class EnhancedNeuralNetwork(NeuralNetwork):
    def __init__(self, layer_sizes=None, input_size=None):
        if layer_sizes is None:
            # Add dropout layers and more hidden units
            layer_sizes = [input_size, 128, 64, 32, len(intents_list)]
        super().__init__(layer_sizes, input_size)
        self.dropout_rate = 0.3
        
    def forward(self, x):
        """Forward pass with dropout"""
        activations = [x]
        zs = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(w, activations[-1]) + b
            zs.append(z)
            
            if i == len(self.weights)-1:  # output layer
                activation = self.softmax(z)
            else:  # hidden layers
                activation = self.relu(z)
                # Apply dropout during training
                if hasattr(self, 'is_training') and self.is_training:
                    mask = np.random.binomial(1, 1-self.dropout_rate, size=activation.shape) 
                    activation *= mask / (1-self.dropout_rate)
            
            activations.append(activation)
        return activations, zs

class EnhancedChatbot:
    """Main chatbot class with all functionality"""
    def __init__(self):
        self.vocab_manager = VocabularyManager()
        self.load_intents()
        self.processor = ChatbotProcessor()
        self.conversation_history = []
        self.context = None
        self.context_expiry = 3
        self.context_counter = 0
        self.weather_api_key = os.getenv('WEATHER_API_KEY')
        self.test_results = []
        self.news_api = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))
        self.dictionary_api_url = "https://api.dictionaryapi.dev/api/v2/entries/en/"

        # Try to load existing vocabulary
        if self.vocab_manager.load_vocabulary():
            unigrams, bigrams = self.vocab_manager.get_vocabulary()
            self.processor.vocabulary = unigrams
            self.processor.bigram_vocab = bigrams
        else:
            # Build new vocabulary only if needed
            self._build_vocabulary()
            
        # Initialize network with correct size
        input_size = len(self.processor.vocabulary) + len(self.processor.bigram_vocab)
        self.layer_sizes = [input_size, 64, len(self.intents_list)]
        self.nn = EnhancedNeuralNetwork(layer_sizes=self.layer_sizes)
        
        # Load or train model
        self._load_or_train_model()
        
    def load_or_build_vocabulary(self):
        """Load existing vocabulary or build new one"""
        vocab_path = 'vocabulary.pkl'
        
        if os.path.exists(vocab_path):
            try:
                with open(vocab_path, 'rb') as f:
                    vocab_data = pickle.load(f)
                    self.processor.vocabulary = vocab_data['unigrams']
                    self.processor.bigram_vocab = vocab_data['bigrams']
                    return True
            except Exception as e:
                logging.error(f"Error loading vocabulary: {e}")
        
        # Build new vocabulary if loading fails
        self._build_vocabulary()
        
        # Save vocabulary
        vocab_data = {
            'unigrams': self.processor.vocabulary,
            'bigrams': self.processor.bigram_vocab,
            'timestamp': datetime.now().isoformat()
        }
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab_data, f)
        
        return False

    def _load_or_train_model(self):
        """Smart model loading/training"""
        model_path = 'chatbot_model.pkl'
        
        if os.path.exists(model_path):
            try:
                self.nn.load_model(model_path)
                # Validate model architecture matches current vocabulary size
                expected_input_size = len(self.processor.vocabulary) + len(self.processor.bigram_vocab)
                if self.nn.layer_sizes[0] != expected_input_size:
                    print(f"Model input size ({self.nn.layer_sizes[0]}) doesn't match current vocabulary size ({expected_input_size}). Retraining...")
                    self.train_model(epochs=50)
                return
            except Exception as e:
                logging.error(f"Error loading model: {e}")
                # Train only if necessary
                self.train_model(epochs=50)
        else:
            # Train model if not found
            self.train_model(epochs=50)
        
    def _build_vocabulary(self):
        """Build vocabulary consistently"""
        all_patterns = [p for intent in self.intents.values() for p in intent['patterns']]
        augmented_data = self.augment_training_data(self.intents)
        self.processor.build_vocabulary(all_patterns + [p for p, _ in augmented_data])
        
        # Save vocabulary metadata
        vocab_metadata = {
            'unigrams': self.processor.vocabulary,
            'bigrams': self.processor.bigram_vocab,
            'timestamp': datetime.now().isoformat()
        }
        with open('vocabulary.pkl', 'wb') as f:
            pickle.dump(vocab_metadata, f)

    def _initialize_network(self):
        """Initialize network with current vocabulary size"""
        input_size = len(self.processor.vocabulary) + len(self.processor.bigram_vocab)
        layer_sizes = [input_size, 64, len(self.intents_list)]
        self.nn = EnhancedNeuralNetwork(layer_sizes=layer_sizes)
        print(f"Initialized new network (input size: {input_size})")

    def load_intents(self):
        """Load intents from JSON file"""
        # with open('intents.json') as f:
        #     self.intents = json.load(f)
        self.intents = intents
        self.intents_list = list(self.intents.keys())

    def train_model(self, epochs=100, learning_rate=0.05, batch_size=32):
        """Optimized training with early stopping"""
        # First ensure consistent vocabulary
        self._build_vocabulary()

        print("Starting training process...")
        
        # Prepare training data in batches
        all_patterns = [p for intent in self.intents.values() for p in intent['patterns']]
        self.processor.build_vocabulary(all_patterns)  # Now build full vocabulary
        
        # Reinitialize network with final size
        input_size = len(self.processor.vocabulary) + len(self.processor.bigram_vocab)
        layer_sizes = [input_size, 64, len(self.intents_list)]
        self.nn = EnhancedNeuralNetwork(layer_sizes=layer_sizes)
        
        # Create training data
        training_data = []
        for intent_name, intent_data in self.intents.items():
            for pattern in intent_data['patterns']:
                vector = self.processor.text_to_vector(pattern)
                label = self._intent_to_onehot(intent_name)
                training_data.append((vector, label))
        
        # Add augmented data
        augmented_data = self.augment_training_data(self.intents)
        for pattern, intent_name in augmented_data:
            vector = self.processor.text_to_vector(pattern)
            label = self._intent_to_onehot(intent_name)
            training_data.append((vector, label))
        
        # Shuffle and batch the data
        random.shuffle(training_data)
        batches = [training_data[i:i + batch_size] 
                  for i in range(0, len(training_data), batch_size)]
        
        # Training loop with early stopping
        best_loss = float('inf')
        patience = 3
        patience_counter = 0
        losses = []
        
        self.nn.is_training = True
        start_time = datetime.now()
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in batches:
                batch_loss = 0
                for x, y in batch:
                    activations, zs = self.nn.forward(x)
                    grad_w, grad_b = self.nn.backward(activations, zs, y)
                    self.nn.update_parameters(grad_w, grad_b, learning_rate)
                    batch_loss += self.nn.compute_loss(activations[-1], y)
                
                epoch_loss += batch_loss / len(batch)
            
            avg_loss = epoch_loss / len(batches)
            losses.append(avg_loss)
            
            # Print progress
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Elapsed: {elapsed:.1f}s")
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                self.nn.save_model('chatbot_model.pkl')  # Save best model
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Reduce learning rate if loss plateaus
            if epoch > 10 and losses[-2] - losses[-1] < 0.001:
                learning_rate *= 0.9
                print(f"Reducing learning rate to {learning_rate:.5f}")
        
        self.nn.is_training = False
        print(f"Training completed in {(datetime.now() - start_time).total_seconds():.1f} seconds")
        
        # Quick evaluation
        test_cases = [
            ("hello", "greeting"),
            ("what's the weather", "weather"),
            ("goodbye", "farewell")
        ]
        correct = 0
        for text, true_intent in test_cases:
            result = self.classify_intent(text)
            if result['intent'] == true_intent:
                correct += 1
        print(f"Quick test accuracy: {correct/len(test_cases):.1%}")
        
        return losses
    
    def _intent_to_onehot(self, intent_name):
        """Convert intent name to one-hot vector"""
        onehot = np.zeros(len(self.intents_list))
        onehot[self.intents_list.index(intent_name)] = 1
        return onehot.reshape(-1, 1)

    def classify_intent(self, text):
        # """Classify user input to an intent"""
        # vector = self.processor.text_to_vector(text)
        # activations, _ = self.nn.forward(vector)
        # output = activations[-1].flatten()
        
        # # Get top 3 intents and their confidence scores
        # top_indices = np.argsort(output)[-3:][::-1]
        # top_intents = [(self.intents_list[i], output[i]) for i in top_indices]
        
        # # Apply confidence thresholds
        # primary_intent, primary_confidence = top_intents[0]

        # # If confidence is low, check if second choice is close
        # if primary_confidence < 0.4 and len(top_intents) > 1:
        #     secondary_intent, secondary_confidence = top_intents[1]
        #     if primary_confidence - secondary_confidence < 0.2:
        #         return {
        #             'intent': 'uncertain',
        #             'confidence': primary_confidence,
        #             'top_intents': top_intents,
        #             'message': f"I'm not sure if you meant '{primary_intent}' or '{secondary_intent}'"
        #         }
        
        # # If confidence is very low
        # if primary_confidence < 0.2:
        #     return {
        #         'intent': 'unknown',
        #         'confidence': primary_confidence,
        #         'top_intents': top_intents,
        #         'message': "I didn't understand that. Could you rephrase?"
        #     }
        
        # return {
        #     'intent': primary_intent,
        #     'confidence': primary_confidence,
        #     'top_intents': top_intents
        # }
        # """Main intent classification entry point"""
        # # First get basic classification
        # basic_result = self._basic_intent_classification(text)
        
        # # Apply context awareness
        # final_result = self.classify_intent_with_context(text, basic_result)
        
        # # If uncertain, generate clarification message
        # if final_result['intent'] == 'uncertain':
        #     clarification = self._generate_clarification(final_result['top_intents'])
        #     final_result['message'] = clarification
            
        # return final_result
        """Enhanced intent classification with context awareness"""
        vector = self.processor.text_to_vector(text)
        activations, _ = self.nn.forward(vector)
        output = activations[-1].flatten()
        
        # Get top 3 intents
        top_indices = np.argsort(output)[-3:][::-1]
        top_intents = [(self.intents_list[i], output[i]) for i in top_indices]
        primary_intent, primary_confidence = top_intents[0]
        
        # Apply context awareness
        if self.context:
            context_intent = next((i for i, data in self.intents.items() 
                                 if data.get('context') == self.context), None)
            if context_intent and primary_intent == context_intent:
                primary_confidence = min(1.0, primary_confidence * 1.3)
        
        # Confidence thresholding
        if primary_confidence < 0.4:
            if len(top_intents) > 1 and (primary_confidence - top_intents[1][1]) < 0.2:
                return {
                    'intent': 'uncertain',
                    'confidence': primary_confidence,
                    'top_intents': top_intents,
                    'message': self._generate_clarification(top_intents)
                }
            return {
                'intent': 'unknown',
                'confidence': primary_confidence,
                'message': "I didn't understand that. Could you rephrase?"
            }
        
        return {
            'intent': primary_intent,
            'confidence': primary_confidence,
            'top_intents': top_intents
        }

    # def process_batch(self, queries):
    #     """Process multiple queries efficiently"""
    #     results = []
    #     vectors = [self.processor.text_to_vector(q) for q in queries]
        
    #     # Batch forward pass
    #     for query, vector in zip(queries, vectors):
    #         # Use existing classify_intent and generate_response
    #         intent_result = self.classify_intent(query)
    #         response = self.generate_response(query)
    #         results.append({
    #             'query': query,
    #             'intent': intent_result,
    #             'response': response
    #         })
                
    #     return results
    def process_batch(self, queries):
        """Process multiple queries efficiently by batching the forward pass"""
        results = []
        # Convert all queries to vectors at once
        vectors = [self.processor.text_to_vector(q) for q in queries]
        
        # Do a single forward pass for all vectors
        for query, vector in zip(queries, vectors):
            # Use the pre-computed vector for intent classification
            activations, _ = self.nn.forward(vector)
            output = activations[-1].flatten()
            
            # Get intent classification using the vector
            top_indices = np.argsort(output)[-3:][::-1]
            top_intents = [(self.intents_list[i], output[i]) for i in top_indices]
            primary_intent, primary_confidence = top_intents[0]
            
            # Generate response based on the classified intent
            intent_result = {
                'intent': primary_intent,
                'confidence': primary_confidence,
                'top_intents': top_intents
            }
            
            response = self.generate_response(query)
            
            results.append({
                'query': query,
                'intent': intent_result,
                'response': response,
                'confidence': primary_confidence
            })
                
        return results

    def get_translation(self, text, target_language):
        """Get translation using Google Translate API"""
        try:
            if not text or not target_language:
                return None

            url = "https://api.6thbridge.com/utils/v1/translations/"
            
            data = {
                "text": text,
                "fromLanguage": 'auto',
                "toLanguage": language_mapping.get(target_language),
            }

            headers = {
                "Client-Id": os.getenv('SIXTH_BRIDGE_CLIENT_ID'),
                "Content-Type": "application/x-www-form-urlencoded"
            }

            response = requests.post(url, data=data, headers=headers)
            data = response.json()

            if response.status_code == 200:
                return {
                    'original_text': text,
                    'translated_text': data['data']['result'],
                    'target_language': target_language
                }
            elif response.status_code == 404:
                logging.warning(f"Target language not found: {target_language}")
                return {'error': 'target_language_not_found'}
            else:
                logging.error(f"Translation API error: {data.get('message', 'Unknown error')}")
                return {'error': 'api_error'}
            
        except requests.exceptions.Timeout:
            logging.error("Translation API request timed out")
            return {'error': 'timeout'}
        except Exception as e:
            logging.error(f"Translation API error: {str(e)}")
            return None

    def get_weather(self, location):
        """Get weather data from OpenWeather API"""
        try:
            location = re.sub(r'\b(in|at|near|around|for|of)\b', '', location, flags=re.IGNORECASE).strip()
        
            if not location:
                return None

            base_url = "http://api.openweathermap.org/data/2.5/weather?"
            params = {
                'q': location, 
                'appid': self.weather_api_key,
                'units': 'metric'
            }

            url = base_url + urllib.parse.urlencode(params)
            
            response = requests.get(url)
            data = response.json()

            if response.status_code == 200:
                return {
                    'temperature': data['main']['temp'],
                    'description': data['weather'][0]['description'],
                    'humidity': data['main']['humidity'],
                    'wind': data['wind']['speed'],
                    'location': data['name']
                }
            elif response.status_code == 404:
                logging.warning(f"Location not found: {location}")
                return {'error': 'location_not_found'}
            else:
                logging.error(f"Weather API error: {data.get('message', 'Unknown error')}")
                return {'error': 'api_error'}
            
        except requests.exceptions.Timeout:
            logging.error("Weather API request timed out")
            return {'error': 'timeout'}
        except Exception as e:
            logging.error(f"Weather API error: {str(e)}")
            return None

    def generate_response(self, user_input):
        """Generate response based on intent and context"""
        # Classify intent
        intent_result = self.classify_intent(user_input)

        # Handle uncertain/unknown intents
        if intent_result['intent'] in ['uncertain', 'unknown']:
            return intent_result.get('message', "I'm not sure how to respond to that.")

        intent = intent_result['intent']
        confidence = intent_result['confidence']

        print(f"Intent: {intent}, Confidence: {confidence:.2f}")
        
        # Reset context for greetings
        if intent == "greeting":
            self.context = None
            self.context_counter = 0

        # Check context expiry
        if self.context and self.context_counter >= self.context_expiry:
            self.context = None
            self.context_counter = 0
        
        # Extract entities
        entities = self.processor.extract_entities(user_input)

        # Log interaction
        self._log_interaction(user_input, intent, confidence, entities)

        # Handle low confidence
        if confidence < 0.3:
            return random.choice(self.intents['fallback']['responses'])

        # Get intent data
        intent_data = self.intents[intent]

        # Handle & Update context if needed
        if intent_data.get('context'):
            self.context = intent_data['context']
            self.context_counter = 0
        elif self.context:
            self.context_counter += 1
        
        # Handle specific intents
        if intent == "weather":
            return self.handle_weather_intent(intent_data, entities)
        elif intent == "time":
            current_time = datetime.now().strftime("%I:%M %p")
            return random.choice(intent_data['responses']).format(time=current_time)
        elif intent == "date":
            current_date = datetime.now().strftime("%A, %B %d, %Y")
            return random.choice(intent_data['responses']).format(date=current_date)
        elif intent == "news":
            return self.handle_news_intent(intent_data, entities, user_input)
        elif intent == "user_name":
            return self.handle_user_name_intent(intent_data, user_input)
        elif intent == "search":
            return self.handle_search_intent(intent_data, entities, user_input)
        elif intent == "schedule_meeting":
            return random.choice(intent_data['responses'])
        elif intent == "reminder":
            return self.handle_reminder_intent(intent_data, entities, user_input)
        elif intent == "calculator":
            return self.handle_calculator_intent(intent_data, user_input)
        elif intent == "definition":
            return self.handle_definition_intent(intent_data, entities, user_input)
        elif intent == "recommend":
            return self.handle_recommend_intent(intent_data, entities, user_input)
        elif intent == "price":
            return self.handle_price_intent(intent_data, entities, user_input)
        elif intent == "translate":
            return self.handle_translate_intent(intent_data, user_input)
        else:
            return random.choice(intent_data['responses'])

    def extract_news_topic(self, user_input):
        """Extract news topic from user input"""
        patterns = [
            r'show me news about (.*)',
            r'latest news on (.*)',
            r'what\'?s new about (.*)',
            r'news about (.*)',
            r'recent articles on (.*)',
            r'tell me about (.*)',
            r'find news on (.*)',
            r'get updates about (.*)',
            r'what\'?s happening with (.*)',
            r'search news for (.*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def get_news_articles(self, topic, page_size=3):
        """Get news articles from API with error handling"""
        try:
            response = self.news_api.get_everything(
                q=topic,
                sources='bbc-news,the-verge',
                language='en',
                sort_by='publishedAt',
                page_size=page_size
            )
            
            if response['status'] == 'ok' and response['totalResults'] > 0:
                return response['articles']
            return None
            
        except Exception as e:
            logging.error(f"News API request failed: {str(e)}")
            return None

    def format_date(self, date_str):
        """Format published date for better readability"""
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%SZ')
            return date_obj.strftime('%b %d, %Y at %I:%M %p')
        except:
            return date_str  

    def handle_news_intent(self, intent_data, entities, user_input):
        """Handle news requests and format API response"""
        # Extract topic from user input
        topic = self.extract_news_topic(user_input)
        
        if not topic:
            return "What topic would you like news about? For example: 'Show me news about technology'"
        
        try:
            # Get news articles
            articles = self.get_news_articles(topic)
            
            if not articles:
                return f"I couldn't find any recent news about {topic}. Try another topic."
            
            # Format the response
            response = random.choice(intent_data['responses']).format(topic=topic) + "\n\n"
            
            # Add up to 3 articles
            for i, article in enumerate(articles[:3]):
                response += f"{i+1}. {article['title']}\n"
                response += f"   Source: {article['source']['name']}\n"
                response += f"   Published: {self.format_date(article['publishedAt'])}\n"
                response += f"   {article['url']}\n\n"
            
            return response.strip()
        
        except Exception as e:
            logging.error(f"News API error: {str(e)}")
            return "I'm having trouble accessing news right now. Please try again later."

    def handle_user_name_intent(self, intent_data, user_input):
        """Handle user name introduction"""
        # Extract name from patterns like "My name is John" or "I am John"
        name = None
        patterns = [
            r'my name is (\w+)',
            r'i am (\w+)',
            r'call me (\w+)',
            r'i\'m called (\w+)',
            r'my friends call me (\w+)',
            r'name\'s (\w+)',
            r'i go by (\w+)',
            r'they call me (\w+)',
            r'it\'s (\w+)',
            r'the name is (\w+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                name = match.group(1)
                break
        
        if name:
            return random.choice(intent_data['responses']).format(name=name)
        else:
            return "Nice to meet you! What's your name?"

    def handle_search_intent(self, intent_data, entities, user_input):
        """Handle search requests"""
        # Extract topic from patterns like "Find information on AI"
        topic = None
        patterns = [
            r'find information on (.*)',
            r'search for (.*)',
            r'look up (.*)',
            r'i want to know about (.*)',
            r'can you find (.*)',
            r'tell me about (.*)',
            r'search (.*)',
            r'find (.*)',
            r'information on (.*)',
            r'details about (.*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                topic = match.group(1).strip()
                break
        
        if topic:
            return random.choice(intent_data['responses']).format(topic=topic)
        else:
            return "What would you like me to search for?"

    def handle_reminder_intent(self, intent_data, entities, user_input):
        """Handle reminder requests"""
        # Extract task from patterns like "Remind me to call mom"
        task = None
        patterns = [
            r'remind me to (.*)',
            r'set a reminder to (.*)',
            r'don\'t let me forget to (.*)',
            r'i need to remember to (.*)',
            r'can you remind me to (.*)',
            r'alert me when (.*)',
            r'notification for (.*)',
            r'ping me about (.*)',
            r'remember to (.*)',
            r'set an alarm for (.*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                task = match.group(1).strip()
                break
        
        if task:
            return random.choice(intent_data['responses']).format(task=task)
        else:
            return "What would you like me to remind you about?"

    def handle_calculator_intent(self, intent_data, user_input):
        """Handle basic calculations"""
        try:
            # Extract math expression
            expr = re.sub(r'calculate|what\'?s|solve|compute|math problem|arithmetic|add|subtract|multiply|divide', 
                         '', user_input, flags=re.IGNORECASE)
            expr = expr.replace('?', '').strip()
            
            # Basic safety check
            if any(c in expr for c in ['import', 'exec', 'eval', 'open']):
                return "I can't perform that calculation for security reasons."
            
            # Evaluate the expression
            result = eval(expr)
            return random.choice(intent_data['responses']).format(result=result)
        except:
            return "I couldn't calculate that. Please provide a valid math expression."

    def extract_definition_term(self, user_input):
        """Extract term to define from user input"""
        patterns = [
            r'define (?:the word |the term )?(.*)',
            r'what is the meaning of (.*)',
            r'what does (.*) mean',
            r'explain the term (.*)',
            r'give me a definition of (.*)',
            r'meaning of (.*)',
            r'what\'?s the definition of (.*)',
            r'describe the term (.*)',
            r'what is (.*)',
            r'explain what is (.*)',
            r'dictionary definition of (.*)',
            r'define the word (.*)',
            r'tell me about the word (.*)',
            r'what\'?s meant by (.*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                term = match.group(1).strip()
                # Remove any trailing question marks or spaces
                term = term.rstrip('?').strip()
                return term.lower()
        return None
    
    def get_word_definition(self, term):
        """Get word definition from dictionary API"""
        try:
            response = requests.get(f"{self.dictionary_api_url}{term}")
            if response.status_code == 200:
                print(f"Dictionary API response: {response.json()}")

                return response.json()
            return None
        except Exception as e:
            logging.error(f"Dictionary API request failed for '{term}': {str(e)}")
            return None

    def format_definition_response(self, intent_data, term, definition_data):
        """Format the dictionary API response into a readable message"""
        if not definition_data or len(definition_data) == 0:
            return f"I couldn't find a definition for '{term}'."
        
        first_entry = definition_data[0]
        response = random.choice(intent_data['responses']).format(term=term)
      
        # Add phonetic pronunciation if available
        if 'phonetic' in first_entry:
            response += f"\n\nPronunciation: {first_entry['phonetic']}"
        
        # Add origin if available
        if 'origin' in first_entry and first_entry['origin']:
            response += f"\n\nOrigin: {first_entry['origin']}"
        
        # Add meanings
        if 'meanings' in first_entry and first_entry['meanings']:
            response += "\n\nMeanings:"
            for meaning in first_entry['meanings']:
                response += f"\n\n{meaning['partOfSpeech'].capitalize()}:"
                for i, definition in enumerate(meaning['definitions'][:3]):  # Limit to 3 definitions per part of speech
                    response += f"\n{i+1}. {definition['definition']}"
                    if 'example' in definition and definition['example']:
                        response += f"\n   Example: {definition['example']}"
        
        print(f"Formatted definition response: {response}")
        return response

    def handle_definition_intent(self, intent_data, entities, user_input):
        """Handle word definition requests using dictionary API"""
        term = self.extract_definition_term(user_input)
        
        if not term:
            return "What word would you like me to define? For example: 'Define hello' or 'What does altruism mean?'"
        
        try:
            definition_data = self.get_word_definition(term)
            
            if not definition_data:
                return f"I couldn't find a definition for '{term}'. Please check the spelling or try another word."
            
            return self.format_definition_response(intent_data, term, definition_data)
        
        except Exception as e:
            logging.error(f"Dictionary API error: {str(e)}")
            return "I'm having trouble accessing dictionary definitions right now. Please try again later."

    def handle_recommend_intent(self, intent_data, entities, user_input):
        """Handle recommendation requests"""
        # Extract topic for recommendations
        topic = None
        patterns = [
            r'recommend (.*)',
            r'suggest (.*)',
            r'what\'?s good for (.*)',
            r'i need ideas for (.*)',
            r'can you recommend (.*)',
            r'give me suggestions for (.*)',
            r'what do you suggest for (.*)',
            r'ideas for (.*)',
            r'best options for (.*)',
            r'top picks for (.*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                topic = match.group(1).strip()
                break
        
        if topic:
            # In a real implementation, you'd generate recommendations based on the topic
            suggestions = f"some great options related to {topic}"  # Replace with actual recommendation logic
            return random.choice(intent_data['responses']).format(topic=topic, suggestions=suggestions)
        else:
            return "What would you like recommendations for?"

    def handle_price_intent(self, intent_data, entities, user_input):
        """Handle price inquiries"""
        # Extract item to price check
        item = None
        patterns = [
            r'how much does (.*) cost',
            r'price of (.*)',
            r'how much is (.*)',
            r'what\'?s the price of (.*)',
            r'cost of (.*)',
            r'pricing for (.*)',
            r'how expensive is (.*)',
            r'price range for (.*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                item = match.group(1).strip()
                break
        
        if item:
            return random.choice(intent_data['responses']).format(item=item)
        else:
            return "What item would you like price information for?"

    def handle_translate_intent(self, intent_data, user_input):
        """Handle translation requests"""
        # Extract phrase and target language
        patterns = [
            r'translate (.*) (?:to|in) (\w+)',
            r'how do you say (.*) (?:in|in the) (\w+)',
            r'in (\w+), how do you say (.*)'
        ]
        
        phrase = None
        language = None
        
        for pattern in patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                if len(match.groups()) == 2:
                    phrase = match.group(1).strip()
                    language = match.group(2).strip().capitalize()
                break
        
        if phrase and language:
            response = self.get_translation(phrase, language)
            if response and 'error' not in response:
                translation = response['translated_text']
                return f"In {language}, you would say: {translation}"
            elif response and response['error'] == 'target_language_not_found':
                return random.choice([
                    f"I couldn't find a translation for {phrase} in {language}. Please check the language code.",
                    f"Sorry, I don't support translations to {language}. Try a different language.",
                    f"Translation to {language} isn't available. Please specify another language."
                ])
        else:
            return "I can help with translations. Please specify a phrase and language (e.g., 'How do you say hello in Spanish?')"
    
    def handle_weather_intent(self, intent_data, entities):
        """Special handling for weather requests"""
        location = entities.get('location', ['unknown location'])[0]
        location = location.split(',')[0].replace('?', '').strip()
        
        weather_data = self.get_weather(location)

        print(f"Weather data for {location}: {weather_data}")
  
        if weather_data and 'error' not in weather_data:
            return random.choice(intent_data['responses']).format(
                location=location,
                humidity=weather_data.get('humidity', 'unknown'),
                temp=weather_data.get('temperature', 'unknown'),
                wind=weather_data.get('wind', 'unknown'),
                description=weather_data.get('description', 'unknown'),
            )
        elif weather_data and weather_data['error'] == 'location_not_found':
            return random.choice([
                f"I couldn't find weather information for {location}. Could you try a nearby city?",
                f"Sorry, I don't have weather data for {location}. Try specifying a major city.",
                f"Weather data for {location} isn't available. Please try a different location."
            ])
        return random.choice([
            f"Sorry, I couldn't get weather data for {location}"
            f"I'm having trouble accessing weather data  for {location}. Please try again later.",
            f"Weather service is currently unavailable for {location}. My apologies!",
            f"I can't reach the weather service at the moment. Maybe ask me something else?"
        ])
    
    def _log_interaction(self, user_input, intent, confidence, entities):
        """Log conversation details"""
        interaction = {
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'intent': intent,
            'confidence': confidence,
            'entities': entities,
            'context': self.context
        }
        self.conversation_history.append(interaction)
        logging.info(json.dumps(interaction))

    def augment_training_data(self, intents):
        """Generate synthetic training examples"""
        augmented_data = []
        synonyms = {
            "hello": ["hi", "hey", "greetings"],
            "goodbye": ["bye", "see you", "farewell"],
            "thanks": ["thank you", "appreciate it", "much obliged"],
            "weather": ["forecast", "temperature", "conditions"],
            "news": ["headlines", "updates", "reports"]
        }
        
        for intent_name, intent_data in intents.items():
            for pattern in intent_data['patterns']:
                # Original pattern
                augmented_data.append((pattern, intent_name))
                
                # Synonym replacement
                for word, syns in synonyms.items():
                    if word in pattern.lower():
                        for syn in syns:
                            new_pattern = pattern.lower().replace(word, syn)
                            augmented_data.append((new_pattern, intent_name))
                
                # Question/statement variations
                if '?' not in pattern:
                    augmented_data.append((pattern + '?', intent_name))
                else:
                    augmented_data.append((pattern.replace('?', ''), intent_name))
                
                # Add common prefixes
                prefixes = ["Can you", "Could you", "Please", "I need to"]
                for prefix in prefixes:
                    augmented_data.append((f"{prefix} {pattern.lower()}", intent_name))
        
        return augmented_data

    def create_test_cases(self):
        """Create test cases from intents with some adversarial examples"""
        test_cases = []
        
        # Add regular test cases
        for intent_name, intent_data in self.intents.items():
            for pattern in intent_data['patterns']:
                test_cases.append((pattern, intent_name))
        
        # Add some edge cases
        edge_cases = [
            ("hi there how are you", "greeting"),
            ("bye for now", "farewell"),
            ("thx", "thanks"),
            ("what's the forecast", "weather"),
            ("tell me headlines", "news"),
            ("", "fallback"),  # Empty input
            ("12345", "fallback"),  # Numbers
            ("@#$%^", "fallback")  # Special chars
        ]
        
        test_cases.extend(edge_cases)
        random.shuffle(test_cases)
        return test_cases
    
    def classify_intent_with_context(self, text):
        """Intent classification that considers conversation history"""
        """Apply context-aware adjustments"""
        if self.context:
            context_intent = next((i for i, data in self.intents.items() 
                                 if data.get('context') == self.context), None)
            
            if context_intent:
                # Boost context-relevant intents
                if basic_result['intent'] == context_intent:
                    basic_result['confidence'] = min(1.0, basic_result['confidence'] * 1.3)
                
                # Check for continuation phrases
                if self._is_context_continuation(text):
                    return {
                        'intent': context_intent,
                        'confidence': 0.9,
                        'is_context_continuation': True,
                        'context': self.context
                    }
        
        # Apply confidence thresholds
        primary_intent, primary_confidence = basic_result['intent'], basic_result['confidence']
        
        if primary_confidence < 0.4:
            return {
                'intent': 'unknown',
                'confidence': primary_confidence,
                'message': "I didn't understand that. Could you rephrase?"
            }
        elif primary_confidence < 0.6 and len(basic_result['top_intents']) > 1:
            secondary_intent, secondary_confidence = basic_result['top_intents'][1]
            if primary_confidence - secondary_confidence < 0.2:
                return {
                    'intent': 'uncertain',
                    'confidence': primary_confidence,
                    'top_intents': basic_result['top_intents'],
                    'context': self.context
                }
        
        return basic_result

    def _is_context_continuation(self, text):
        """Check for phrases indicating context continuation"""
        continuation_phrases = [
            "more about", "tell me more", "what else",
            "another", "next", "continue", "go on"
        ]
        return any(phrase in text.lower() for phrase in continuation_phrases)

    def _generate_clarification(self, top_intents):
        """Generate clarification message for uncertain intents"""
        options = [f"'{intent}' ({confidence:.0%})" 
                  for intent, confidence in top_intents[:2]]
        return f"I'm not sure if you meant {options[0]} or {options[1]}. Can you clarify?"

    def evaluate_intent_recognition(self, test_cases):
        """Evaluate intent recognition performance"""
        results = {
            'correct': 0,
            'incorrect': 0,
            'uncertain': 0,
            'confidence_scores': [],
            'confusion_matrix': defaultdict(lambda: defaultdict(int))
        }
        
        for text, true_intent in test_cases:
            result = self.classify_intent(text)
            
            if result['intent'] == 'uncertain':
                results['uncertain'] += 1
            elif result['intent'] == true_intent:
                results['correct'] += 1
                if 'confidence' in result:
                    results['confidence_scores'].append(result['confidence'])
            else:
                results['incorrect'] += 1
                results['confusion_matrix'][true_intent][result['intent']] += 1
        
        # Calculate metrics
        total = len(test_cases)
        results['accuracy'] = results['correct'] / total if total > 0 else 0
        results['uncertainty_rate'] = results['uncertain'] / total if total > 0 else 0
        results['avg_confidence'] = np.mean(results['confidence_scores']) if results['confidence_scores'] else 0
        
        # Log confusion matrix
        logging.info("Confusion Matrix:")
        for true_intent, predictions in results['confusion_matrix'].items():
            logging.info(f"True: {true_intent} -> Predicted: {dict(predictions)}")
        
        return results

    def evaluate_model(self, test_data):
        """Evaluate model performance on test data"""
        y_true = []
        y_pred = []

        for text, true_intent in test_data:
            result = self.classify_intent(text)
            y_true.append(self.intents_list.index(true_intent))
            y_pred.append(self.intents_list.index(result['intent']))

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'timestamp': datetime.now().isoformat()
        }

        self.test_results.append(metrics)
        logging.info(f"Model evaluation: {json.dumps(metrics)}")

        return metrics
    


bot = EnhancedChatbot()
