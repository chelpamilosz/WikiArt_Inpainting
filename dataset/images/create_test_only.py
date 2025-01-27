import pandas as pd
import h5py

def create_test_h5_and_annotations(dataset_h5_path, annotations_csv_path, output_h5_path, output_csv_path):
    df = pd.read_csv(annotations_csv_path)
    test_df = df[df['set_type'] == 'test'].reset_index(drop=True)
    test_df['h5_index'] = test_df.index
    with h5py.File(dataset_h5_path, 'r') as src_h5, h5py.File(output_h5_path, 'w') as dst_h5:
        src_images = src_h5['image']
        dst_images = dst_h5.create_dataset('image', shape=(len(test_df),) + src_images.shape[1:], dtype=src_images.dtype)
        for i, idx in enumerate(test_df['h5_index']):
            dst_images[i] = src_images[idx]
    test_df.to_csv(output_csv_path, index=False)

if __name__ == '__main__':
    create_test_h5_and_annotations('dataset.h5', 'annotations.csv', 'test_dataset.h5', 'test_annotations.csv')