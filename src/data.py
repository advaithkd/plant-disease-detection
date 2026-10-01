from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import train_test_split

SEED = 42

def is_valid(path):
    try:
        data = tf.io.read_file(path)
        tf.io.decode_image(data, channels=3, expand_animations=False)
        return True
    except Exception:
        return False

def get_datasets(data_dir="../data/PlantVillage", img_size=(224, 224), batch=32):
    data_dir = Path(data_dir)
    class_names = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    paths, labels = [], []
    for i, c in enumerate(class_names):
        for f in (data_dir / c).glob("*"):
            if is_valid(str(f)):
                paths.append(str(f))
                labels.append(i)
            else:
                print("Skipping bad file:", f)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)

    def load(path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, img_size)
        return img, label

    def make_ds(p, l, training=False):
        ds = tf.data.Dataset.from_tensor_slices((p, l))
        if training:
            ds = ds.shuffle(len(p), seed=SEED)
        ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(batch).prefetch(tf.data.AUTOTUNE)

    return (make_ds(X_train, y_train, True), make_ds(X_val, y_val),
            make_ds(X_test, y_test), class_names)