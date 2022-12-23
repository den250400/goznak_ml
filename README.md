# MEL-spectrograms classification and denoising
### Установка всех необходимых библиотек
```
pip3 install -r requirements.txt
```

### Обучение, тестирование, и inference
Обучение модели классификации
```
python3 train_classification.py --epochs=50 --dataset_path='./data' --model_filename='classifier.pth'
```
Обучение denoising-модели
```
python3 train_denoising.py --epochs=10 --dataset_path='./data' --model_filename='denoiser.pth'
```
Тестирование модели классификации, выводит на экран accuracy на выбранном датасете
```
python3 train_denoising.py --dataset_path='./data/val' --model_filename='classifier.pth'
```
Тестирование denoising-модели, выводит на экран средний MSE на выбранном датасете
```
python3 train_denoising.py --dataset_path='./data/val' --model_filename='denoiser.pth'
```
