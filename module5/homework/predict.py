import pickle

model_file = 'model1.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

with open("dv.bin", 'rb') as f_in:
    dv = pickle.load(f_in)

def predict(customer):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    loan = y_pred >= 0.5
    
    # print(y_pred)
    result = {
        'loan_probability': float(y_pred),
        'loan': bool(loan)
    }
    return result


client = {"job": "retired", "duration": 445, "poutcome": "success"}

print(predict(client))