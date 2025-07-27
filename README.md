# VisionNet 🧠

A user-friendly desktop application for training and evaluating image classification models, built with PyTorch and PyQt6.

Originally designed to classify fruits and vegetables as part of my master's thesis, VisionNet has been generalized to support any image classification dataset. It provides an intuitive graphical interface that simplifies the process of loading datasets, training various neural network models, and assessing their performance.


## Screenshots 📸 

*In the **Model tab**, you can select a neural network, configure its hyperparameters, and start the training process. Afterwards, the tab displays key metrics, including loss history and a confusion matrix plot.*
![Model Tab - Simple CNN](./assets/images/models-tab-simple-cnn.png)

*This screenshot shows the VGG16 model, the largest available. It also demonstrates the responsive UI, where the sidebar automatically hides when the window is resized.*
![Model Tab - VGG16](./assets/images/models-tab-vgg16.png)

*In the **Classification tab**, you can classify images from the test set (navigating with arrow keys) or upload your own. The predicted class and a probability distribution chart are displayed for each image.*
![Classification Tab](./assets/images/classification-tab.png)

*The **Dataset tab** allows you to load and manage your image classification dataset.*
![Dataset Tab](./assets/images/dataset-tab.png)

*The **Settings tab** currently allows you to switch between light and dark themes.*
![Settings Tab](./assets/images/settings-tab.png)


## Key Features ✨

VisionNet is designed with a strong focus on user experience, providing a seamless workflow for deep learning practitioners and enthusiasts.

*   **Intuitive Graphical Interface:** A clean, modern, and easy-to-navigate UI built with PyQt6.
*   **Effortless Dataset Management:** Easily load any dataset thats intended for image classification. The application expects a standard `train/test/valid` directory structure.
*   **Multiple Model Support:** Comes with several models ready for training:
    *   `ResNet18`
    *   `EfficientNet-B0`
    *   `VGG16`
    *   `SimpleCNN` (a custom baseline model)
*   **Interactive Model Training:** Train models on your own datasets with customizable hyperparameters (learning rate, epochs, momentum).
*   **Save and Load:** Save your trained model weights, calculated metrics, plots and training parameters and load them later for further analysis.
*   **Metrics and Visualization:** Analyze the model using metrics and plots.
*   **Classification:** Use the trained models to classify images.


## Installation ⚙️

Follow these steps to run the application on your local machine.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/filip-ba/vision-net.git
    cd VisionNet
    ```

2.  **Install dependencies:**
    Ensure you have Python (version 3.10+ recommended) and pip installed.
    ```bash
    pip install -r requirements.txt
    ```

3. **To start the application, use the following command:**
    ```bash
    python main.py
    ```


## Future Roadmap 🔮

Here are some features planned for future releases:

*   **Custom Neural Network Builder:** Instead of relying only on a `SimpleCNN`, users will be able to design their own neural network architectures through the UI, meaning they could configure parameters like:
    *   Number and type of layers (Convolutional, Pooling, Fully Connected)
    *   Number of filters, kernel size, stride
    *   Activation functions (ReLU, Sigmoid, etc.)
    *   Dropout rates
*   **Expanded Model Library:** Another useful thing would be to integrate more CNN models from the pytorch library. Users would then be able to select which models they want to work with via a settings panel, keeping the interface clean and tailored to their needs.
*   **Code Refactoring:** Ongoing efforts to refactor the codebase for better performance, scalability, and maintainability.


## 📝 License

This project is not under any license.