from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
data = [
    ("headache fever cough", "Common Cold"),
    ("fever chills fatigue", "Flu"),
    ("rash itching redness", "Allergy"),
    ("rash itching stomach pain", "Drug reaction"),
    ("fatigue restlessness obesity", "Diabetese"),
    ("Cough Fatigue Breathlessness", "Asthama"),
    ("Headache dizziness Chest Pain", "Hypertension"),
    ("rash itching redness", "Allergy"),
    ("High Fever Yellow skin Vomiting", "Jaundice"),
    ("High fever Chills Nausea muscle pain ", "Malaria"),
    ("Mild fever Skinrash Itching", "Chicken-Pox"),
    ("Highfever Red Spots Nausia", "Dengue"),
    ("Diarrhoea High fever Vomiting", "Typhoid"),
    ("Breathlessness Cough Sweating", "Tuberculosis"),
    (" joint pain vomiting yellowish skin dark urine nausea","hepatitis A"),
    ("continuous sneezing shivering watering from eyes","Allergy"),
    ("stomach pain ulcers on tongue vomiting cough chest pain","GERD"),
    (" vomiting loss of appetite abdominal pain passage of gases","Peptic ulcer diseae"),
    (" muscle wasting patches in throat high fever","AIDS"),
    (" itching nodal skin eruptions dischromic patches","Fungal infection"),
    ("itching skin rash dischromic patches","Fungal infection"),
    (" continuous sneezing shivering chills watering_from_eyes","Allergy"),
    (" acidity indigestion headache excessive hunger stiff neck depression","Migraine"),
    (" chills fatigue cough high fever breathlessness sweating malaise chest pain fast heart rate rusty sputum","Pneumonia"),
    (" constipation pain during bowel movements pain in anal region bloody stool irritation in anus","Dimorphic hemmorhoids(piles)"),
    (" muscle weakness stiff neck swelling joints movement stiffness painful walking painful walking","Arthritis"),
    (" vomiting headache nausea spinning movements loss of balance unsteadiness","(vertigo) Paroymsal  Positional Vertigo"),
    ("burning micturition bladder discomfort continuous feel of urine","Urinary tract infection"),
    ("skin rash joint pain skin peeling silver like dusting small dents in nails inflammatory nails","Psoriasis"),
    ("vomiting indigestion loss of appetite abdominal pain passage of gases internal itching","Peptic ulcer diseae"),
    ("muscle wasting patches in throat high fever extra marital contacts","AIDS"),
    ("fatigue weight loss restlessness lethargy irregular sugar level blurred and distorted vision obesity","Diabetes"),
    ("itching skin rash fatigue lethargy high fever headache loss of appetite mild fever swelled lymph nodes malaise","Chicken pox"),
    ("constipation pain during bowel movements pain in anal region bloody stool irritation in anus","Dimorphic hemmorhoids(piles)"),
    ("fatigue weight gain cold hands and feets mood swings lethargy dizziness puffy face and eyes","Hypothyroidism"),
    ("fatigue mood swings weight loss restlessness sweating diarrhoea fast heart rate excessive hunger","Hyperthyroidism"),
    ("vomiting fatigue anxiety sweating headache nausea blurred and distorted vision excessive hunger","Hypoglycemia"),
    ("joint pain neck pain knee pain hip joint pain swelling joints painful walking","Osteoarthristis"),
    ("vomiting headache nausea spinning movements loss of balance unsteadiness","(vertigo) Paroymsal  Positional Vertigo"),
    ("skinrash pus filled pimples blackheads scurring","Acne"),
    ("chills fatigue cough high fever breathlessness sweating malaise phlegm chest pain fast heart rate","Pneumonia"),
    ("joint pain vomiting fatigue high fever yellowish skin dark urine nausea loss of appetite abdominal pain","Hepatitis E"),
    ("itching vomiting fatigue weight loss high fever yellowish skin dark urine,","Jaundice"),
    ("itching fatigue lethargy yellowish skin dark urine loss of appetite abdominal pain yellow urine yellowing of eyes","Hepatitis B"),
    ("fatigue yellowish skin nausea loss of appetite yellowing of eyes  family history","Hepatitis C"),
    ("joint pain vomiting fatigue yellowish skin dark urine nausea loss of appetite","Hepatitis D"),
    (" fatigue cough high fever breathlessness","Bronchial Asthma"),
    (" vomiting sunken eyes dehydration diarrhoea","Gastroenteritis"),
    (" skin rash joint pain skin peeling silver like dusting","Psoriasis"),
    (" skin rash chills joint pain vomiting high fever headache nausea loss of appetite","Dengue"),
    (" back pain weakness in limbs neck pain  dizziness","Cervical spondylosis"),
    (" fatigue cramps bruising obesity swollen legs","Varicose veins"),
    (" vomiting headache altered sensorium","Paralysis (brain hemorrhage),"),
    (" vomiting yellowish skin abdominal pain swelling of stomach distention of abdomen","Alcoholic hepatitis"),
    (" skin rash blister red sore around nose yellow crust ooze","Impetigo"),
    (" fatigue cough high fever breathlessness mucoid sputum","Bronchial Asthma"),
    (" chills joint pain vomiting fatigue high fever headache nausea","Dengue"),
    (" chills vomiting fatigue weight loss cough high fever breathlessness","Tuberculosis"),
    (" continuous sneezing chills fatigue high fever headache","Common Cold"),
    (" breathlessness sweating chest pain","Heart attack"),
    (" vomiting fatigue anxiety sweating headache nausea excessive hunger","Hypoglycemia"),
    (" joint pain neck pain knee pain hip joint pain","Osteoarthristis"),
    (" burning micturition bladder discomfort foul smell of urine","Urinary tract infection"),
    (" muscle wasting patches in throat high fever extra marital contact","AIDS"),
    (" fatigue high fever breathlessness family history mucoid sputum","Bronchial Asthma"),
    (" fatigue cough high fever breathlessness sweating","Pneumonia"),
    (" vomiting breathlessness sweating chest pain","Heart attack"),
    (" fatigue weight gain cold hands and feets mood swings","HypothyroidismY"),
    ("vomiting fatigue anxiety sweating headache nausea","Hypoglycemia "),
    ("joint pain neck pain knee pain hip joint pain","Osteoarthristis "),
    (" skin rash blister red sore around nose","Impetigo"),
    (" skin rash joint pain skin peeling silver like dusting","Psoriasis"),
    (" itching skin rash nodal skin eruptions","Fungal infection"),
    (" stomach pain acidity ulcers on tongue vomiting cough chest pain","GERD"),
    (" stomach pain acidity ulcers on tongue cough chest pain","GERD"),
    (" itching skin rash burning micturition spotting urination","Drug Reaction"),
    (" skin rash stomach pain burning micturition spotting urination","Drug Reaction"),
    (" disease vomiting indigestion loss of appetite abdominal pain","Peptic ulcer"),
    ("  fatigue weight loss restlessness lethargy","Diabetes "),
    (" fatigue weight loss restlessness lethargy irregular sugar level","Diabetes "),
    (" vomiting sunken eyes dehydration diarrhoea","Gastroenteritis"),
    ("sunken eyes dehydration diarrhoea","Gastroenteritis "),
    ("  headache chest pain dizziness loss of balance","Hypertension"),
    (" headache chest pain loss of balance","Hypertension"),
    (" acidity indigestion headache blurred and distorted vision excessive hunger","Migraine"),
    (" itching vomiting weight loss high fever  yellowish skin dark urine","Jaundice"),
    (" vomiting high fever sweating headache nausea diarrhoea muscle pain","Malaria"),
    (" itching skin rash fatigue lethargy high fever headache","Chicken pox"),
    (" chills joint pain vomiting fatigue high fever headache nausea","Dengue"),
    (" joint pain vomiting yellowish skin dark urine nausea abdominal pain","hepatitis A"),
    (" itching lethargy yellowish skin dark urine loss of appetite","Hepatitis B"),
    (" joint pain vomiting fatigue yellowish skin dark urine","Hepatitis D"),
    (" chills vomiting fatigue weight loss cough breathlessness sweating","Tuberculosis"),
    ("continuous sneezing chills fatigue cough high fever headache","Common Cold "),
    (" vomiting breathlessness sweating chest pain","Heart attack"),
    (" chills vomiting fatigue high fever headache nausea constipation","Typhoid"),
    (" chills vomiting high fever sweating headache nausea diarrhoea muscle_pain","Malaria"),




    
]

symptoms, labels = zip(*data)
model = make_pipeline(CountVectorizer(), DecisionTreeClassifier())
model.fit(symptoms, labels)
def predict_disease(symptoms_input):
    predicted_label = model.predict([symptoms_input])
    return predicted_label[0]
user_input = input("\n Hello!! I am your HealthBuddy.\n Do you feel unwell ?? :Y/N ")
if(user_input=='Y' ):

    user_input = input("Enter few symptoms you feel separated by spaces: ")
    predicted_disease = predict_disease(user_input)
    print(f"The predicted disease based on symptoms is: {predicted_disease}")
else:
    print("Thank You. I wish you well!!")