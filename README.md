# neGpt: Nepali Language Generation (Character Level)

neGpt is a character-level language generation model specifically designed for the Nepali language. It utilizes a Nepali news dataset to train using the concepts of "Attention Is All You Need" research paper in machine learning and generate text at the character level. While the current model does a pretty good job at generating text character by character, it has limitations in capturing the relationships between words due to its character-level approach.

Future enhancements aim to train the model at the token level and incorporate more advanced algorithms to better understand and represent word relationships.

---

## Dataset Information

- **Source**: [Nepali News Dataset (Large)](https://github.com/pemagrg1/Nepali-Datasets?tab=readme-ov-file)
- **Uploader**: [Ashok Pant on Kaggle](https://www.kaggle.com/datasets/ashokpant/nepali-news-dataset-large)
- **Dataset Character Length**: 15,215,541
- **Vocabulary Size**: 90
- **Character List**: ['\t', '\n', '\x1e', '\x1f', ' ', '\xa0', 'ँ', 'ं', 'ः', 'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ए', 'ऐ', 'ऑ', 'ओ', 'औ', 'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ऱ', 'ल', 'व', 'श', 'ष', 'स', 'ह', '़', 'ऽ', 'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'ॆ', 'े', 'ै', 'ॉ', 'ो', 'ौ', '्', 'ॐ', '॒', 'क़', 'ड़', 'ढ़', 'फ़', 'य़', '।', '०', '१', '२', '३', '४', '५', '६', '७', '८', '९', '॰', '\u202f']

---

## Model Metrics Obtained

- **Training Loss**: 1.2948
- **Validation Loss**: 1.4472

---

## How to Use

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/neGpt.git
   cd neGpt
   ```

2. **Set up a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  
   # On Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Update configuration parameters**:
   - Update any necessary parameters (e.g., dataset paths, model hyperparameters, model paths).

5. **Run the main script to start the model**:
   ```bash
   python main.py
   ```