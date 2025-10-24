```text
 ____   __       __     ____   ____ 
(  _ \ (  )     /__\   (_   ) ( ___)
 ) _ <  )(__   /(__)\   / /_   )__) 
(____/ (____) (__)(__) (____) (____)

```


---

# Stereo Depth Reconstruction Pipeline System Architecture

**Author:** Blaze
**Date:** *25 Oct 2025*

---

## Abstract

This technical document provides a comprehensive description of a stereo matching pipeline system architecture. The system implements a complete workflow from web-based data acquisition to trained neural network models for disparity estimation. The pipeline is organized into four modular components with well-defined interfaces and data flow. Each module handles specific responsibilities including data collection, preprocessing, feature extraction, and model training.

---

## 1. System Architecture Overview

The stereo matching pipeline is designed as a sequential processing system with four interconnected modules. The architecture follows a data-flow pattern where each module consumes the output of the previous module and produces input for the next.

---

## 2. Module 1: Data Acquisition System

### 2.1 Module Purpose and Scope

The data acquisition module is responsible for automatically retrieving the **Middlebury 2014 stereo dataset** from the official web repository. It handles web scraping, file discovery, directory structure generation, and robust download management.

### 2.2 Core Algorithm Implementation

#### Algorithm: Data Acquisition Main Workflow

```text
Procedure AcquireDataset(base_url, download_path)
    Initialize web session with browser headers
    ScanForZipFiles(base_url)
    FilterDatasets(file_list, exclude_keywords)
    GenerateSiteStructure(dataset_list)
    For each dataset in dataset_list:
        DownloadCoreFiles(dataset_url, download_path)
        DownloadExposureVariants(dataset_url, download_path)
    Return VerifyDownloads(download_path)
```

### 2.3 Key Component Details

#### (a) Web Scraping Engine – File Discovery Algorithm

```text
Function FindDownloadables(url, extensions, filenames)
    response ← HTTP_GET(url)
    soup ← BeautifulSoup(response.content)
    downloadables ← []
    For each anchor in soup:
        href ← extract href attribute
        If href valid:
            absolute_url ← urljoin(base_url, href)
            If url matches extensions or filenames:
                downloadables.append({filename, link})
    Return downloadables
```

#### (b) Directory Structure Generator – Site Structure Generation

```text
Function GenerateOrgSite(zip_files)
    org_sites ← []
    For each zip_file in zip_files:
        base_path ← replace_extension(zip_file.link, '')
        dataset_path ← replace_substring(base_path, 'zip', 'datasets')
        org_sites.append(dataset_path + '/')
        For level in ['L1','L2','L3','L4']:
            org_sites.append(dataset_path + '/ambient/' + level + '/')
    Return org_sites
```

#### (c) File Download Manager – Robust File Download

```text
Function DownloadFiles(file_list, download_location, org_site)
    For each file_info in file_list:
        file_url ← file_info['link']
        full_path ← join(download_location, relative_path)
        If not exists(full_path):
            create_directories(dirname(full_path))
            stream_download(file_url, full_path)
            log_success(filename)
        Else:
            log_skip(filename)
```

### 2.4 Data Structures and Output

This module produces an organized directory structure containing:

* **Stereo Images:** `im0.png`, `im1.png`
* **Disparity Maps:** `disp0.pfm`, `disp1.pfm`
* **Calibration Files:** `calib.txt`
* **Exposure Variants:** Multiple illumination conditions per scene

---

## 3. Module 2: Data Preprocessing System

### 3.1 Purpose and Scope

Transforms raw downloaded data into standardized, cleaned formats suitable for machine learning. Handles file parsing, image processing, data augmentation, and serialization.

### 3.2 Core Processing Pipeline

```text
Procedure PreprocessData(raw_path, output_path, resize_factor)
    calibration_data ← ProcessAllCalibrations(raw_path)
    image_data ← ProcessAllImages(raw_path, resize_factor)
    disparity_data ← ProcessAllDisparities(raw_path, resize_factor)
    SerializeData(calibration_data, image_data, disparity_data, output_path)
```

### 3.3 Component Algorithms

#### (a) Calibration Parser

```text
Function LoadCalibration(file_path)
    calib ← {}
    For each line in file:
        If line contains '=':
            Parse key, value
            If value is matrix: parse as array
            Else: convert to int/float
    Return calib
```

#### (b) Disparity Map Processor

```text
Function LoadPFMDisparity(file_path)
    Parse header, width, height, scale
    data ← read binary floats
    disparity ← reshape(data)
    disparity ← flip vertically
    Return CleanDisparity(disparity)
```

#### (c) Disparity Cleaning Algorithm

```text
Function CleanArray(arr, max_disp)
    result ← where(arr < 0, 0, arr)
    Replace inf with max_val
    Normalize if max_val > max_disp
    Return result as int16
```

#### (d) Image Processing Pipeline

```text
Function ProcessImage(image_path, resize_factor)
    rgb_array ← load RGB image
    gray_array ← convert to grayscale
    Resize if needed
    Return uint8 array
```

#### (e) Data Augmentation through Flipping

```text
Procedure GenerateAugmentedData(images, disparities)
    For each original image pair:
        AddOriginal()
        AddFlipped()
```

---

## 4. Module 3: Patch Generation and Feature Extraction

### 4.1 Purpose and Scope

Generates training samples by extracting image patches and computing feature descriptors. Creates corresponding patch-strip pairs across the disparity range with feature validation.

### 4.2 Core Generation Algorithm

```text
Procedure GeneratePatchesStrips(output_path, patch_shape, target_samples, params)
    For i in 0..target_samples:
        FindValidPatch()
        If valid:
            ExtractPatchStripPair()
            ComputeAndStoreFeatures()
```

### 4.3 Patch Sampling Algorithm

```text
Function FindValidPatch(sample_index, patch_shape, params)
    Repeat until valid or max_tries:
        SampleRandomLocation()
        If ValidatePatch(): return patch_location
    Return None
```

### 4.4 Feature Extraction System

```text
Function ComputeFeatures(patch)
    mean_val ← mean(patch)
    std_val ← std(patch)/0.5
    skew_val ← normalized skewness
    dct_mean ← mean(abs(DCT(patch)))
    entropy ← -Σ p*log2(p)
    sobel_x, sobel_y ← mean gradients normalized
    Return [mean_val, std_val, skew_val, dct_mean, entropy, sobel_x, sobel_y]
```

### 4.5 Feature Validation System

```text
Function FeatureError(feat1, feat2)
    diff ← abs(feat1 - feat2)
    Return any(diff > thresholds)
```

### 4.6 Strip Extraction Algorithm

```text
Function ExtractStrip(right_image, patch_coords, disparity_range, params)
    For d in disparities:
        x_start ← shift by disparity
        strip[d] ← cropped region
        features[d] ← ComputeFeatures(strip[d])
    Return strip, features
```

---

## 5. Module 4: Neural Network Training System

### 5.1 Purpose and Scope

Implements and trains a **siamese neural network** for disparity estimation using generated patches and features. Handles model definition, training, and evaluation.

### 5.2 Model Architecture Definition

```text
Function CreateStereoModel(patch_shape, feature_dim, max_d)
    Define left and right inputs
    Left branch: Conv2D → Dense
    Right branch: TimeDistributed encoder
    Compute cosine similarity → Softmax
    Return Model(inputs, outputs)
```

### 5.3 Custom Layers

#### (a) Similarity Layer

Computes cosine similarity between left and right embeddings with masking.

```text
Class SimilarityLayer(Layer)
    Call(inputs):
        Normalize embeddings
        similarity ← dot(E_left, E_right)
        Apply mask
        Return similarity / 0.1
```

#### (b) Clip Layer

Ensures numerical stability.

```text
Class ClipLayer(Layer)
    Call(inputs):
        Return clip(inputs, 1e-7, 1.0)
```

### 5.4 Data Generator Implementation

Efficiently loads training batches from memory-mapped arrays.

```text
Class StereoBatchGenerator(Sequence)
    __getitem__(index):
        Load patches, features, masks
        Subsample disparities if needed
        Return model_inputs, labels
    on_epoch_end():
        Shuffle indices
```
