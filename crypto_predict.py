import torch
from ModelTrainer import SimpleTransformer, Trainer, train_model, load_model, TransformerModel, EnhancedLSTMModel
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Cryptocurrency Price Prediction')
    subparsers = parser.add_subparsers(dest='command')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data-file', default='preprocessed_data.pt')
    train_parser.add_argument('--epochs', type=int, default=50)
    train_parser.add_argument('--batch-size', type=int, default=32)
    train_parser.add_argument('--no-gpu', action='store_true')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--data-file', default='preprocessed_data.pt')
    predict_parser.add_argument('--output-file', default='predictions.csv')
    predict_parser.add_argument('--no-gpu', action='store_true')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        trainer, scaler_y, crypto_ids = train_model(
            args.data_file,
            args.epochs,
            args.batch_size,
            not args.no_gpu
        )
        print("Training completed. Model saved as 'best_model.pt'")
    
    elif args.command == 'predict':
        data = torch.load(args.data_file)
        model = SimpleTransformer(input_size=data['X_test'].shape[2])
        model.load_state_dict(torch.load('best_model.pt'))
        
        trainer = Trainer(model, not args.no_gpu)
        test_loader = DataLoader(TensorDataset(data['X_test'], data['y_test']), batch_size=32)
        predictions, _ = trainer.evaluate(test_loader)
        
        # Inverse transform predictions
        predictions = data['scaler_y'].inverse_transform(torch.tensor(predictions).numpy())
        
        # Save results
        results_df = pd.DataFrame({
            'crypto_id': data['crypto_ids'],
            'prediction': predictions.flatten()
        })
        results_df.to_csv(args.output_file, index=False)
        print(f"Predictions saved to {args.output_file}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()