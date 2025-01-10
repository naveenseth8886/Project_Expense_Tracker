from flask import Flask, request, render_template
import pickle

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')



@app.route('/budget.html', methods = ['GET','POST'])
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
        print(city,occ,dep,inc,dss,age,lr,ins)
        
        # Call the predict function with the user inputs
        prediction = predict(city, occ, dep, inc, age, lr, ins, rent)
        
        # You can then pass the prediction to the template if needed
        return render_template('budget.html', prediction=prediction)
    else:
        return render_template('budget.html')

def predict(city, occupation, dependents, income, age, loan_repayment, insurance, rent):
    ## Model Loded Here
    Groceries_model = pickle.load(open('Model/Groceries.pkl','rb'))
    Eating_out_model = pickle.load(open('Model/Eating_Out.pkl','rb'))
    Education_model = pickle.load(open('Model/education_liReg_model.pkl','rb'))
    Entertainment_model = pickle.load(open('Model/Entertainment.pkl','rb'))
    Health_care_model = pickle.load(open('Model/Healthcare.pkl','rb'))
    Miscellaneous_care_model = pickle.load(open('Model/Miscellaneous.pickle','rb'))
    Transport_care_model = pickle.load(open('Model/Transport.pkl','rb'))
    Utilities_care_model = pickle.load(open('Model/Utilities.pkl','rb'))
    
    ## City Encode
    Groceries_city_encode = pickle.load(open('City/Groceries_lb2_City.pkl','rb'))
    Eating_out_city_encode = pickle.load(open('City/Eating_Out_City.pkl','rb'))
    Education_city_encode = pickle.load(open('City/education_City.pkl','rb'))
    Entertainment_city_encode = pickle.load(open('City/Entertainment_city.pkl','rb'))
    Health_care_city_encode = pickle.load(open('City/Healthcare_City.pkl','rb'))
    Miscellaneous_care_city_encode = pickle.load(open('City/Miscellaneous_city.pickle','rb'))
    Transport_care_city_encode = pickle.load(open('City/Transport_city.pkl','rb'))
    Utilities_care_city_encode = pickle.load(open('City/Utilities_city.pkl','rb'))
    
    ## Occupation Encode
    Groceries_occ_encode = pickle.load(open('Occupation/Groceries_lb1_occupation.pkl','rb'))
    Eating_out_occ_encode = pickle.load(open('Occupation/Eating_Out_Occupation.pkl','rb'))
    Education_occ_encode = pickle.load(open('Occupation/education_Occupation.pkl','rb'))
    Entertainment_occ_encode = pickle.load(open('Occupation/Entertainment_occupation.pkl','rb'))
    Health_care_occ_encode = pickle.load(open('Occupation/Healthcare_Occupation (1).pkl','rb'))
    Miscellaneous_care_occ_encode = pickle.load(open('Occupation/Miscellaneous_occupation.pickle','rb'))
    Transport_care_occ_encode = pickle.load(open('Occupation/Transport_Occupation.pkl','rb'))
    Utilities_care_occ_encode = pickle.load(open('Occupation/Utilities_Occupation.pkl','rb'))

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
    print(Groceries_result)
    return Groceries_result
predict("Tier_1","Self_Employed","2","50000","25","5000","2000","20000")

## predict("Mumbai","Private","2","50000","10000","25","5000","2000")



if __name__ == '__main__':
    app.run(debug=True, port=5000,host='0.0.0.0')