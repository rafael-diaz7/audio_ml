import json
import math
import os
import librosa

DATASET_PATH = os.path.join("data", "genres_original")
JSON_PATH = os.path.join("data", "data.json")

SAMPLE_RATE = 22050
DURATION = 30  # in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, n_segments=5):
    # dictionary to store data
    data = {
        "mapping": [],  # mapping genre to num
        "mfcc": [],  # training inputs
        "labels": []  # outputs
    }

    n_samples_per_segment = int(SAMPLES_PER_TRACK / n_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(n_samples_per_segment / hop_length)

    # loop through all genres in dataset
    for index, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure that we are not at the root level
        if dirpath is not dataset_path:
            # save the semantic label of folder [classical, blues, ...] for mappings
            dirpath_components = dirpath.split("\\")  # ['data', 'genres_original', 'blues']
            semantic_label = dirpath_components[-1]
            data["mapping"].append(semantic_label)
            print(f"\nProcessing: {semantic_label}")

            # process files for a specific genre
            for f in filenames:
                if f == 'jazz.00054.wav':
                    continue
                file_path = os.path.join(dirpath, f)

                # load audio file
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # process segments extracting mfcc and storing data
                for s in range(n_segments):
                    start_sample = n_samples_per_segment * s  # s=0 -> 0
                    finish_sample = start_sample + n_samples_per_segment  # s=0 -> num_samples_per_segment

                    mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                                sr=sr,
                                                n_mfcc=n_mfcc,
                                                n_fft=n_fft,
                                                hop_length=hop_length)

                    mfcc = mfcc.T  # transpose

                    # store mfcc for segment if it has the expected length (expected_num_mfcc_per_sample)
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(index-1)
                        print(f"{file_path}, segment:{s+1}")
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, n_segments=10)
