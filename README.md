# **ASL to Text Conversion — Architecture Comparison**

## **Overview**
American Sign Language (ASL) is a visually expressive language widely used by the Deaf and Hard of Hearing community. Translating ASL gestures into structured written text presents significant challenges due to its unique grammar and syntax.

In this project, we evaluate and compare the performance of two state-of-the-art natural language processing (NLP) models — **BART** and **T5** — for converting ASL gloss into coherent English sentences. Our focus is on understanding how the size of each architecture affects performance on this task. 

Using the **ASLG-PC12 dataset**, which contains annotated ASL glosses paired with ground-truth English translations, we train and fine-tune different model variants (three sizes of T5 and BART). The goal is to assess how the model size impacts translation quality and accuracy.

Our methodology includes:
- **Model Training & Fine-tuning**: We train BART and T5 models using PyTorch. 
- **Performance Analysis**: We compare models both **qualitatively and quantitatively** on standard NLP metrics.


## **Project Objectives**
- **Analyze Architecture Size**: Study how different sizes of T5 (small, base, large) and BART affect ASL-to-text conversion.
- **ASL Gloss to Text Conversion**: Translate ASL gloss into coherent, human-readable English.
- **Qualitative and Quantitative Analysis**: Measure model performance using standard NLP metrics.

## **Installation**

Follow the steps below to set up the repository on your local machine.

   ```bash
   git clone https://github.com/atsatvik/ASL2Text.git
   cd ASL2Text
   pip install -r requirements.txt
  ```

## **Training**

### **Train T5 Model**
To train a T5 model, use the following command. This script currently supports training on a **single GPU or CPU**. You can customize training parameters by modifying the file `config/config_T5.yml`. By default, checkpoints are saved to `results/train_T5/<exp_num>`. Change it by setting the --exp_name arg.

**Run the command:**
```bash
python model_train_T5.py
```
### **Train BART Model**
To train a BART model, use the following command. You can customize training parameters by modifying the file `config/config_BART.yml`. By default, checkpoints are saved to `results/train_BART/<exp_num>`. Change it by setting the --exp_name arg.

**Run the command:**
```bash
CUDA_VISIBLE_DEVICES=0,1 python model_train_BART.py
```


## **Testing**

### **Test T5 Model**
To test the T5 model, use the following command. You can customize testing parameters by modifying the file `config/config_T5.yml`. By default, inference results are saved to `inference_results/test_T5/<exp_num>`. Change it by setting the --exp_name arg.

**Run the command:**
```bash
python model_test_T5.py --resume path/to/checkpoint 
```
### **Test BART Model**
To test the BART model, use the following command. You can customize testing parameters by modifying the file `config/config_BART.yml`. By default, inference results are saved to `inference_results/test_BART/<exp_num>`. Change it by setting the --exp_name arg.

**Run the command:**
```bash
python model_test_BART.py --resume path/to/checkpoint 
```

## **Results**
### **Quantitative results**
Following are the results of different models on several metrics, computed on the test set.
| **Metrics**         | **T5-Small** | **T5-Base** | **T5-Large** | **BART-Base** | **BART-Large** |
|---------------------|-------------|------------|-------------|---------------|-----------------|
| **BLEU Score**      | 0.818       | 0.880      | **0.893**       | 0.874         | 0.887           |
| **BLEU-1 Score**    | 0.937       | 0.956      | **0.962**       | 0.955         | 0.956           |
| **BLEU-4 Score**    | 0.742       | 0.824      | **0.842**       | 0.817         | 0.830           |
| **ROGUE-1**         | 0.951       | 0.970      | **0.974**       | 0.968         | 0.969           |
| **ROGUE-2**         | 0.882       | 0.923      | **0.930**       | 0.919         | 0.923           |
| **ROGUE-L**         | 0.950       | 0.969      | **0.973**       | 0.967         | 0.968           |
| **ROGUE-LSum**      | 0.950       | 0.969      | **0.973**      | 0.967         | 0.968           |
| **Perplexity**      | 9.410       | 11.099     | 9.219       | 4.503         |**3.937**          |

#### **Validation curve for different metrics**
![Figure_1](https://github.com/user-attachments/assets/3de381d8-befc-4891-823f-9669c9855540)




### Qualitative Results

#### Example 1:
**Gloss:** 
`ON DESC-OR HAND , ONLINE GAME CHANCE BE DESC-OFTEN PLAY OUTSIDE TERRITORY DESC-SINGLE COUNTRY .`

**Ground Truth:** 
`on the other hand , online games of chance are often played outside the territory of a single country .`

**Prediction:** 
`on the other hand , online gaming chances are often played outside the territory of a single country .`

---

#### Example 2:
**Gloss:** 
`DEPLOYMENT THIS MISSION BE DESC-NOT DESC-EASY TO ACHIEVE , AS X-YOU DESC-PROBABLY OBSERVE .`

**Ground Truth:** 
`deployment of this mission was not easy to achieve , as you probably observed .`

**Prediction:** 
`the deployment of this mission was not easy to achieve , as you probably observed .`

---

#### Example 3:
**Gloss:** 
`EUROPEAN PARLIAMENT X-POSS ROLE BE TO PROVIDE DESC-ACTIVE INPUT INTO NATURE AND CONTENT AGREEMENT .`

**Ground Truth:** 
`the european parliament's role is to provide active input into the nature and content of the agreement .`

**Prediction:** 
`the european parliament's role is to provide active input into the nature and content of the agreement .`

---


 






