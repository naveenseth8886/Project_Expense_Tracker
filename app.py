from flask import Flask, request, render_template
import pickle

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')



@app.route('/budget.html', methods= ['GET','POST'])
## Creating Budget Function
def budget():
    if request.method == "POST":
        city = request.form.get("city-tier")
        occ = request.form.get("occupation")
        dep = request.form.get("dependents")
        inc = request.form.get("income")
        dss = request.form.get("desired-savings")
        age = request.form.get("age")
        lr = request.form.get("loan-repayment")
        ins = request.form.get("insurance")
        rent= request.form.get("rent")
        print(city,occ,dep,inc,age,lr,ins,rent)
        
        prediction = predict(city,occ,dep,inc,age,lr,ins,rent)
        
        result = algo_ml(prediction[0],prediction[1],prediction[2],prediction[3],prediction[4],prediction[5],prediction[6],prediction[7],int(dss),int(inc))
        
        return render_template('result.html', result = result)
        
    else:
        return render_template('budget.html')
    

def predict(city,occupation,dependents,income,age,loan_repayment,insurance,rent):
    ## Model Loded Here
    Groceries_model = pickle.load(open(r'Model\Groceries.pkl','rb'))
    Eating_out_model = pickle.load(open(r'Model\Eating_Out.pkl','rb'))
    Education_model = pickle.load(open(r'Model\education_liReg_model.pkl','rb'))
    Entertainment_model = pickle.load(open(r'Model\Entertainment.pkl','rb'))
    Health_care_model = pickle.load(open(r'Model\Health_care_model.pkl','rb'))
    Miscellaneous_care_model = pickle.load(open(r'Model\Miscellaneous.pkl','rb'))
    Transport_care_model = pickle.load(open(r'Model\Transport.pkl','rb'))
    Utilities_care_model = pickle.load(open(r'Model\Utilities.pkl','rb'))
    
    ## City Encode
    Groceries_city_encode = pickle.load(open(r'City\Groceries_lb2_City.pkl','rb'))
    Eating_out_city_encode = pickle.load(open(r'City\Eating_Out_City.pkl','rb'))
    Education_city_encode = pickle.load(open(r'City\Entertainment_city.pkl','rb'))
    Entertainment_city_encode = pickle.load(open(r'City\Entertainment_city.pkl','rb'))
    Health_care_city_encode = pickle.load(open(r'City\Health_care_model_City.pkl','rb'))
    Miscellaneous_care_city_encode = pickle.load(open(r'City\Miscellaneous_City.pkl','rb'))
    Transport_care_city_encode = pickle.load(open(r'City\Transport_City.pkl','rb'))
    Utilities_care_city_encode = pickle.load(open(r'City\Utilities_City.pkl','rb'))
    
    ## Occupation Encode
    Groceries_occ_encode = pickle.load(open(r'Occupation\Groceries_lb1_occupation.pkl','rb'))
    Eating_out_occ_encode = pickle.load(open(r'Occupation\Eating_Out_Occupation.pkl','rb'))
    Education_occ_encode = pickle.load(open(r'Occupation\education_Occupation.pkl','rb'))
    Entertainment_occ_encode = pickle.load(open(r'Occupation\Entertainment_occupation.pkl','rb'))
    Health_care_occ_encode = pickle.load(open(r'Occupation\Health_care_model_Occupation.pkl','rb'))
    Miscellaneous_care_occ_encode = pickle.load(open(r'Occupation\Miscellaneous_Occupation.pkl','rb'))
    Transport_care_occ_encode = pickle.load(open(r'Occupation\Transport_Occupation.pkl','rb'))
    Utilities_care_occ_encode = pickle.load(open(r'Occupation\Utilities_Occupation.pkl','rb'))
    
    def safe_transform(encoder, value):
        try:
            return encoder.transform([value])[0]
        except ValueError as e:
            print(f"Warning: {e}")
            return -1  # or some default value or handling

    Groceries_result = Groceries_model.predict([[
        safe_transform(Groceries_city_encode, city),
        safe_transform(Groceries_occ_encode, occupation),
        int(dependents), int(income), int(age),
        int(loan_repayment), int(insurance), int(rent)
    ]])
    Eating_out_result = Eating_out_model.predict([[
        safe_transform(Eating_out_city_encode, city),
        safe_transform(Eating_out_occ_encode, occupation),
        int(dependents), int(income), int(age),
        int(loan_repayment), int(insurance), int(rent)
    ]])
    Education_result = Education_model.predict([[
        safe_transform(Education_city_encode, city),
        safe_transform(Education_occ_encode, occupation),
        int(dependents), int(income), int(age),
        int(loan_repayment), int(insurance), int(rent)
    ]])
    Entertainment_result = Entertainment_model.predict([[
        safe_transform(Entertainment_city_encode, city),
        safe_transform(Entertainment_occ_encode, occupation),
        int(dependents), int(income), int(age),
        int(loan_repayment), int(insurance), int(rent)
    ]])
    Health_care_result = Health_care_model.predict([[
        safe_transform(Health_care_city_encode, city),
        safe_transform(Health_care_occ_encode, occupation),
        int(dependents), int(income), int(age),
        int(loan_repayment), int(insurance), int(rent)
    ]])
    Miscellaneous_result = Miscellaneous_care_model.predict([[
        safe_transform(Miscellaneous_care_city_encode, city),
        safe_transform(Miscellaneous_care_occ_encode, occupation),
        int(dependents), int(income), int(age),
        int(loan_repayment), int(insurance), int(rent)
    ]])
    Transport_result = Transport_care_model.predict([[
        safe_transform(Transport_care_city_encode, city),
        safe_transform(Transport_care_occ_encode, occupation),
        int(dependents), int(income), int(age),
        int(loan_repayment), int(insurance), int(rent)
    ]])
    Utilities_result = Utilities_care_model.predict([[
        safe_transform(Utilities_care_city_encode, city),
        safe_transform(Utilities_care_occ_encode, occupation),
        int(dependents), int(income), int(age),
        int(loan_repayment), int(insurance), int(rent)
    ]])
    print(Groceries_result,Eating_out_result,Education_result,Entertainment_result,
          Health_care_result,Miscellaneous_result,Transport_result,Utilities_result)
    
    return Groceries_result[0],Eating_out_result[0],Education_result[0],Entertainment_result[0],Health_care_result[0],Miscellaneous_result[0],Transport_result[0],Utilities_result[0]
          
## predict("Tier_1","Self_Employed","2","50000","25","5000","2000","20000")


## preparing the ML algorithm
def algo_ml(Groceries_result,Eating_out_result,Education_result,Entertainment_result,
          Health_care_result,Miscellaneous_result,Transport_result,Utilities_result,Desired_saving,income):
    expense = Groceries_result+Eating_out_result+Education_result+Entertainment_result+Health_care_result+Miscellaneous_result+Transport_result+Utilities_result
    sv = income - expense
    dsv = income*Desired_saving/100
    asv = dsv - sv 
    x = asv/expense
    agr = Groceries_result-(Groceries_result*x)
    aeo = Eating_out_result-(Eating_out_result*x)
    aed = Education_result-(Education_result*x)
    aet = Entertainment_result-(Entertainment_result*x)
    ahc = Health_care_result-(Health_care_result*x)
    amc = Miscellaneous_result-(Miscellaneous_result*x)
    atc = Transport_result-(Transport_result*x)
    auc = Utilities_result-(Utilities_result*x)
    

    
    # store the result in dictionary
    result = {
        "Groceries":agr,
        "Eating_out":aeo,
        "Education":aed,
        "Entertainment":aet,
        "Health_care":ahc,
        "Miscellaneous":amc,
        "Transport":atc,
        "Utilities":auc}
    
    print(result)
    
    return result 
    

if __name__ == '__main__':
    app.run(debug=True, port=5000, host="0.0.0.0")