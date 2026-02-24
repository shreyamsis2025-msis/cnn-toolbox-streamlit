import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
layers = tf.keras.layers
models = tf.keras.models
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import regularizers

import os
import tensorflow as tf

@st.cache_resource
def load_pretrained_model():
    if os.path.exists("cnn_cifar10_demo.h5"):
        return tf.keras.models.load_model("cnn_cifar10_demo.h5")
    return None

# Auto-load model
if "model" not in st.session_state:
    pretrained = load_pretrained_model()
    if pretrained:
        st.session_state["model"] = pretrained

def enlarge(img, scale=8):
    return cv2.resize(
        img,
        (img.shape[1]*scale, img.shape[0]*scale),
        interpolation=cv2.INTER_NEAREST
    )


st.set_page_config(page_title="CNN Toolbox", layout="wide")

# ---------------------------
# Load Dataset
# ---------------------------
@st.cache_data
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    class_names = [
        "airplane","automobile","bird","cat","deer",
        "dog","frog","horse","ship","truck"
    ]
    return x_train, y_train, x_test, y_test, class_names

x_train, y_train, x_test, y_test, class_names = load_data()

# Normalize
x_train = x_train / 255.0
x_test  = x_test  / 255.0

@st.cache_resource
def build_model(filters1, filters2, kernel_size, dropout, lr, l2_val):

    model = tf.keras.Sequential([
        layers.Conv2D(filters1, kernel_size,
                      activation='relu',
                      padding='same',
                      kernel_regularizer=regularizers.l2(l2_val),
                      input_shape=(32,32,3)),
        layers.BatchNormalization(),

        layers.Conv2D(filters1, kernel_size,
                      activation='relu',
                      padding='same',
                      kernel_regularizer=regularizers.l2(l2_val)),
        layers.MaxPooling2D(),
        layers.Dropout(dropout),

        layers.Conv2D(filters2, kernel_size,
                      activation='relu',
                      padding='same',
                      kernel_regularizer=regularizers.l2(l2_val)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(dropout),

        layers.Flatten(),
        layers.Dense(256,
                     activation='relu',
                     kernel_regularizer=regularizers.l2(l2_val)),
        layers.Dropout(dropout),

        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ---------------------------
# Sidebar Menu
# ---------------------------
st.sidebar.title("🧠 CNN Toolbox")
page = st.sidebar.radio(
    "Select Topic",
    ["Dataset Understanding",
     "Train CNN Model",
     "Normalization",
     "Convolution Demo",
     "Stride & Padding",
     "Types of Convolution",
     "Pooling Visualizer",
     "Padding Visualizer",
     "Decision Boundary Visualizer",
     "CNN Animation"
]
)


# ---------------------------
# Page 1
# ---------------------------
if page == "Dataset Understanding":

    st.title("📊 CIFAR-10 Dataset Understanding")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Dataset Shapes")
        st.write("Training images:", x_train.shape)
        st.write("Training labels:", y_train.shape)
        st.write("Test images:", x_test.shape)
        st.write("Test labels:", y_test.shape)

    with col2:
        st.write("### Class Names")
        st.write(class_names)

    st.write("### Sample Images")

    fig, axes = plt.subplots(5,5, figsize=(6,6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(x_train[i])
        ax.set_title(class_names[int(y_train[i])], fontsize=8)
        ax.axis("off")
    st.pyplot(fig)

    st.write("### One Image Info")
    sample = x_train[0]
    st.write("Image Shape:", sample.shape)
    st.write("Pixel Range:", sample.min(), "to", sample.max())

elif page == "Train CNN Model":

    st.title("🚀 Train CNN with Hyperparameter Tuning")

    st.subheader("🔧 Hyperparameters")

    col1, col2 = st.columns(2)

    with col1:
        filters1 = st.slider("Conv1 Filters", 16, 128, 32, step=16)
        filters2 = st.slider("Conv2 Filters", 32, 256, 64, step=32)
        kernel_size = st.selectbox("Kernel Size", [3,5])
        l2_val = st.select_slider(
    "L2 Regularization",
    options=[0.0, 1e-5, 1e-4, 1e-3],
    value=1e-4
)

    with col2:
        dropout = st.slider("Dropout", 0.1, 0.6, 0.25)
        lr = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.0005, 0.001, 0.005],
            value=0.0005
        )
        batch_size = st.select_slider(
            "Batch Size",
            options=[32, 64, 128],
            value=64
        )

    epochs = st.slider("Epochs", 5, 50, 10)

    if st.button("Start Training"):

        model = build_model(filters1, filters2,
                    kernel_size, dropout, lr, l2_val)

        st.write("### 📈 Live Accuracy")

        chart = st.line_chart()

        class StreamlitCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                chart.add_rows({
                    "Train Accuracy": [logs["accuracy"]],
                    "Val Accuracy": [logs["val_accuracy"]]
                })

        history = model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            callbacks=[StreamlitCallback()],
            verbose=1
        )

        st.success("🎉 Training Complete!")

        # -----------------------
        # Final Accuracy
        # -----------------------
        train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
        test_loss, test_acc   = model.evaluate(x_test, y_test, verbose=0)

        st.subheader("📊 Final Accuracy")
        c1, c2 = st.columns(2)

        with c1:
            st.metric("Train Accuracy", f"{train_acc*100:.2f}%")

        with c2:
            st.metric("Test Accuracy", f"{test_acc*100:.2f}%")

        best_val = max(history.history['val_accuracy'])
        st.success(f"🏆 Best Validation Accuracy: {best_val*100:.2f}%")

        # -----------------------
        # Accuracy Graph
        # -----------------------
        st.subheader("📈 Accuracy Graph")
        st.line_chart({
            "Train Accuracy": history.history['accuracy'],
            "Val Accuracy": history.history['val_accuracy']
        })

        # -----------------------
        # Loss Graph
        # -----------------------
        st.subheader("📉 Loss Graph")
        st.line_chart({
            "Train Loss": history.history['loss'],
            "Val Loss": history.history['val_loss']
        })

        st.session_state["model"] = model



elif page == "Normalization":

    st.title("🧹 Image Normalization")

    st.write("""
    ### Why Normalize?
    Neural networks train faster when inputs are small.
    We scale pixel values from **0–255 → 0–1**
    """)

    # Choose image
    idx = st.slider("Choose Image Index", 0, len(x_train)-1, 0)

    original = x_train[idx]
    normalized = original / 255.0

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Original Image")
        st.image(original)
        st.write("Pixel range:", original.min(), "to", original.max())

    with col2:
        st.write("### Normalized Image")
        st.image(normalized)
        st.write("Pixel range:", normalized.min(), "to", normalized.max())

    st.write("---")

    st.subheader("🔬 Try Different Scaling")

    scale = st.slider("Divide pixels by", 1, 255, 255)

    scaled = original / scale

    # Rescale to 0–1 for display
    scaled_display = (scaled - scaled.min()) / (scaled.max() - scaled.min() + 1e-8)

    st.image(scaled_display, caption=f"Image divided by {scale}")
    st.write("New range:", scaled.min(), "to", scaled.max())




elif page == "Convolution Demo":

    st.title("🔍 Convolution Visualizer")

    st.write("""
    Convolution slides a filter over the image to detect patterns.
    Try different filters below.
    """)

    # choose image
    idx = st.slider("Choose Image Index", 0, len(x_train)-1, 0)
    img = x_train[idx]
    img = img.astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    st.image(img, caption="Original Image")

    # filters
    filters = {
        "Edge Detection": np.array([[-1,-1,-1],
                                    [-1, 8,-1],
                                    [-1,-1,-1]]),

        "Blur": np.ones((3,3))/9,

        "Sharpen": np.array([[0,-1,0],
                             [-1,5,-1],
                             [0,-1,0]]),

        "Emboss": np.array([[-2,-1,0],
                            [-1,1,1],
                            [0,1,2]])
    }

    choice = st.selectbox("Choose Filter", list(filters.keys()))
    kernel = filters[choice]

    # st.write("### Kernel Used")
    # st.write(kernel)

    # apply convolution
    output = cv2.filter2D(gray, -1, kernel)

    st.image(output, caption="Feature Map", clamp=True)

    st.write("""
    ### Explanation
    Each pixel is multiplied by kernel values and summed.
    CNN learns these kernels automatically during training.
    """)

    st.set_page_config(layout="wide")

    big_img = enlarge(img, 8)
    st.image(big_img, width=True)

elif page == "Stride & Padding":

    st.title("📏 Stride & Padding Visualizer")

    import cv2

    idx = st.slider("Choose Image Index", 0, len(x_train)-1, 0)
    img = x_train[idx]
    img = img.astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


    stride = st.slider("Stride", 1, 4, 1)
    padding = st.slider("Padding", 0, 5, 0)

    kernel = np.array([[1,0,-1],
                       [1,0,-1],
                       [1,0,-1]])

    # st.write("### Kernel")
    # st.write(kernel)

    # Add padding
    if padding > 0:
        gray = np.pad(gray, padding, mode='constant')

    # Manual convolution with stride
    k = kernel.shape[0]
    out_h = (gray.shape[0]-k)//stride + 1
    out_w = (gray.shape[1]-k)//stride + 1

    output = np.zeros((out_h, out_w))

    for i in range(0, out_h):
        for j in range(0, out_w):
            region = gray[i*stride:i*stride+k,
                          j*stride:j*stride+k]
            output[i,j] = np.sum(region * kernel)

    # Normalize for display
    output = (output - output.min()) / (output.max() - output.min() + 1e-8)

    # Enlarge for clarity
    def enlarge(img, s=6):
        return cv2.resize(img, (img.shape[1]*s, img.shape[0]*s),
                          interpolation=cv2.INTER_NEAREST)

    st.image(enlarge(img,8), caption="Original")
    st.image(enlarge(output,8), caption="Output Feature Map")

    st.write("### Output Shape:", output.shape)

    st.write("""
    **Explanation**
    - Larger stride → smaller output
    - Padding keeps border info
    - CNN learns best stride automatically
    """)


elif page == "Types of Convolution":

    import cv2
    st.title("🧠 Types of Convolution")

    conv_type = st.selectbox(
    "Select Convolution Type",
    ["1x1 Convolution",
     "3x3 Convolution",
     "Depthwise Convolution",
     "Spatially Separable Demo",
     "Standard Conv2D Demo",
     ]
)


    idx = st.slider("Choose Image Index", 0, len(x_train)-1, 0)
    img = x_train[idx]

    # enlarge function
    def enlarge(img, scale=8):
        return cv2.resize(img,
                          (img.shape[1]*scale, img.shape[0]*scale),
                          interpolation=cv2.INTER_NEAREST)

    st.image(enlarge(img), caption="Original Image")

    # ---------------------------
    # 1x1 Convolution
    # ---------------------------
    if conv_type == "1x1 Convolution":

        st.subheader("1×1 Convolution")

        num_filters = st.slider("Number of Filters", 1, 8, 3)

        h, w, c = img.shape
        filters = np.random.randn(num_filters, c)

        outputs = []

        for f in range(num_filters):
            out = np.zeros((h, w))
            for ch in range(c):
                out += img[:,:,ch] * filters[f][ch]
            outputs.append(out)

        outputs = np.stack(outputs, axis=2)

        # normalize for display
        outputs = (outputs - outputs.min())/(outputs.max()-outputs.min()+1e-8)

        # show each feature map
        st.write("### Feature Maps")

        for i in range(num_filters):
            st.image(enlarge(outputs[:,:,i]), caption=f"Filter {i+1}")


    # ---------------------------
    # 3x3 Convolution
    # ---------------------------
    elif conv_type == "3x3 Convolution":

        st.subheader("3x3 Convolution")

        kernel = np.array([[1,0,-1],
                           [1,0,-1],
                           [1,0,-1]])

        img = img.astype(np.float32)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        output = cv2.filter2D(gray, -1, kernel)
        output = (output-output.min())/(output.max()-output.min()+1e-8)

        st.image(enlarge(output), caption="3x3 Output")

        st.write("Standard convolution detecting edges.")

    # ---------------------------
    # Depthwise Convolution
    # ---------------------------
    elif conv_type == "Depthwise Convolution":

        st.subheader("Depthwise Convolution")

        kernel = np.ones((3,3))/9

        outputs = []
        for c in range(3):
            out = cv2.filter2D(img[:,:,c], -1, kernel)
            outputs.append(out)

        output = np.stack(outputs, axis=2)
        output = (output-output.min())/(output.max()-output.min()+1e-8)

        st.image(enlarge(output), caption="Depthwise Output")

        st.write("Each channel is convolved separately.")

    # ---------------------------
    # Spatially Separable Demo
    # ---------------------------
    elif conv_type == "Spatially Separable Demo":

        st.subheader("Spatially Separable Convolution")

        filters = st.slider("Number of Filters", 1, 8, 4)
        kernel_size = st.selectbox("Kernel Size", [3,5])
        idx = st.slider("Image Index", 0, len(x_train)-1, 0)

        img = x_train[idx] / 255.0
        img_batch = img.reshape(1, 32, 32, 3)

        conv1 = layers.Conv2D(
            filters=filters,
            kernel_size=(1, kernel_size),
            activation="relu",
            padding="same"
        )

        conv2 = layers.Conv2D(
            filters=filters,
            kernel_size=(kernel_size, 1),
            activation="relu",
            padding="same"
        )

        x = conv1(img_batch)
        feature_maps = conv2(x).numpy()

        st.image(enlarge(img), caption="Input Image")

        st.write("### Feature Maps")
        cols = st.columns(filters)

        for i in range(filters):
            with cols[i]:
                fm = feature_maps[0, :, :, i]
                fm = (fm - fm.min())/(fm.max()-fm.min()+1e-8)
                st.image(enlarge(fm), caption=f"F{i+1}")

    # ---------------------------
    # Standard Conv2D Demo
    # ---------------------------
    elif conv_type == "Standard Conv2D Demo":

        st.subheader("Standard Conv2D")

        filters = st.slider("Number of Filters", 1, 8, 4)
        kernel_size = st.selectbox("Kernel Size", [3,5])
        idx = st.slider("Image Index", 0, len(x_train)-1, 0)

        img = x_train[idx] / 255.0
        img_batch = img.reshape(1, 32, 32, 3)

        conv = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            activation="relu",
            padding="same"
        )

        feature_maps = conv(img_batch).numpy()

        st.image(enlarge(img), caption="Input Image")

        st.write("### Feature Maps")
        cols = st.columns(filters)

        for i in range(filters):
            with cols[i]:
                fm = feature_maps[0, :, :, i]
                fm = (fm - fm.min())/(fm.max()-fm.min()+1e-8)
                st.image(enlarge(fm), caption=f"F{i+1}")
    



elif page == "Pooling Visualizer":

    st.title("🏊 Pooling Visualizer")

    x_train, y_train, x_test, y_test, class_names = load_data()


    pooling_type = st.selectbox(
        "Pooling Type",
        ["Max Pooling", "Average Pooling"]
    )

    pool_size = st.slider("Pool Size", 2, 4, 2)
    idx = st.slider("Image Index", 0, len(x_train)-1, 0)

    img = x_train[idx] / 255.0
    img_batch = img.reshape(1, 32, 32, 3)

    conv = layers.Conv2D(
        filters=1,
        kernel_size=3,
        activation="relu",
        padding="same"
    )

    feature_map = conv(img_batch)

    if pooling_type == "Max Pooling":
        pool = layers.MaxPooling2D(pool_size=pool_size)
    else:
        pool = layers.AveragePooling2D(pool_size=pool_size)

    pooled = pool(feature_map).numpy()

    fm = feature_map.numpy()[0, :, :, 0]
    fm = (fm-fm.min())/(fm.max()-fm.min()+1e-8)

    pooled_img = pooled[0, :, :, 0]
    pooled_img = (pooled_img-pooled_img.min())/(pooled_img.max()-pooled_img.min()+1e-8)

    st.image(enlarge(img), caption="Input Image")
    st.image(enlarge(fm), caption="Feature Map")
    st.image(enlarge(pooled_img), caption=f"After {pooling_type}")

    st.write("Output Shape:", pooled.shape)



elif page == "Padding Visualizer":

    st.title("🧱 Padding Visualizer")

    x_train, y_train, x_test, y_test, class_names = load_data()


    padding_type = st.selectbox("Padding Type", ["valid", "same"])
    kernel_size = st.selectbox("Kernel Size", [3, 5])
    idx = st.slider("Image Index", 0, len(x_train)-1, 0)

    img = x_train[idx] / 255.0
    img_batch = img.reshape(1, 32, 32, 3)

    conv = layers.Conv2D(
        filters=1,
        kernel_size=kernel_size,
        activation="relu",
        padding=padding_type
    )

    feature_map = conv(img_batch).numpy()

    fm = feature_map[0, :, :, 0]
    fm = (fm-fm.min())/(fm.max()-fm.min()+1e-8)

    st.image(enlarge(img), caption="Input Image")
    st.image(enlarge(fm), caption=f"Padding = {padding_type}")

    st.write(
        f"Input Size: 32×32 → Output Size: "
        f"{feature_map.shape[1]}×{feature_map.shape[2]}"
    )

elif page == "Decision Boundary Visualizer":

    from sklearn.neural_network import MLPClassifier

    st.title("🧠 Decision Boundary Visualizer")

    # -------- Spiral Dataset --------
    def generate_spiral(n_points=200, noise=0.2):
        np.random.seed(0)
        theta = np.sqrt(np.random.rand(n_points)) * 2 * np.pi
        r = 2 * theta + np.pi
        x1 = np.array([np.cos(theta)*r, np.sin(theta)*r]).T
        x2 = np.array([np.cos(theta+np.pi)*r, np.sin(theta+np.pi)*r]).T
        X = np.vstack([x1, x2]) + noise*np.random.randn(2*n_points,2)
        y = np.hstack([np.zeros(n_points), np.ones(n_points)])
        return X, y

    # -------- Sliders --------
    hidden_neurons = st.slider("Hidden Neurons", 0, 100, 0, step=5)
    lr = st.slider("Learning Rate", 0.0001, 0.1, 0.01)
    epochs = st.slider("Epochs", 500, 5000, 2000, step=500)

    X, y = generate_spiral()

    # -------- Model --------
    if hidden_neurons == 0:
        model = MLPClassifier(
            hidden_layer_sizes=(),
            activation='identity',
            learning_rate_init=lr,
            max_iter=epochs
        )
        title = "Linear Classifier"
    else:
        model = MLPClassifier(
            hidden_layer_sizes=(hidden_neurons,),
            activation='tanh',
            learning_rate_init=lr,
            max_iter=epochs
        )
        title = f"Non-Linear Classifier (Hidden Neurons = {hidden_neurons})"

    model.fit(X, y)

    # -------- Decision Boundary --------
    xx, yy = np.meshgrid(
        np.linspace(X[:,0].min()-1, X[:,0].max()+1, 300),
        np.linspace(X[:,1].min()-1, X[:,1].max()+1, 300)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(X[:,0], X[:,1], c=y, s=20)
    ax.set_title(title)
    st.pyplot(fig, width=300)

    # st.pyplot(fig)

    st.write("""
    **Explanation**
    - Hidden neurons allow curved decision boundaries.
    - More neurons → more complex patterns learned.
    - Linear model cannot separate spiral data.
    """)

elif page == "CNN Animation":

    import time
    import tensorflow as tf
    from tensorflow.keras import Model

    st.title("🎬 CNN Working Animation")
    speed = st.slider("Animation Speed", 0.2, 3.0, 1.0)

    if "model" not in st.session_state:
        st.warning("⚠️ Train model first in 'Train CNN Model' page")
        st.stop()

    model = st.session_state["model"]

    idx = st.slider("Choose Image", 0, len(x_train)-1, 0)
    img = x_train[idx] 
    img_batch = img.reshape(1,32,32,3)

    placeholder = st.empty()

    # ---------------------------
    # STEP 1 - Input Image
    # ---------------------------
    placeholder.image(enlarge(img, 10), caption="Step 1: Input Image")
    st.button("Continue Animation")
    time.sleep(speed)

    # ---------------------------
    # STEP 2 - Get REAL Feature Maps
    # ---------------------------
    conv_layers = [l for l in model.layers if isinstance(l, Conv2D)]

    for layer in conv_layers[:1]:  # show first conv layer
        feature_model = Model(inputs=model.inputs,
                              outputs=layer.output)

        feature_maps = feature_model.predict(img_batch)[0]

        for i in range(min(4, feature_maps.shape[-1])):
            fm = feature_maps[:,:,i]
            fm = (fm-fm.min())/(fm.max()-fm.min()+1e-8)
            placeholder.image(enlarge(fm),
                              caption=f"Step 2: Feature Map {i+1}")
            time.sleep(0.8)

    # ---------------------------
    # STEP 3 - Prediction
    # ---------------------------
    pred = model.predict(img_batch)[0]
    predicted_class = class_names[np.argmax(pred)]

    placeholder.write(f"### 🎯 Prediction → {predicted_class}")

    st.bar_chart(pred)
    st.success("🎉 CNN Finished Processing!")
    
    true_class = class_names[int(y_train[idx])]
    st.write(f"Actual → {true_class}")
    
    if predicted_class == true_class:
        st.success("✅ Correct Prediction")
    else:
        st.error("❌ Wrong Prediction")

