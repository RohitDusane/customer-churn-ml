/* ===========================================================
   -----------------------------------------------------------
   ✔ Tab Switching
   ✔ Slider Styling
   ✔ Toggle Buttons
   ✔ Single Prediction (AJAX)
   ✔ Loading Button
   ✔ Error Handling
   =========================================================== */

document.addEventListener("DOMContentLoaded", () => {

    initializeSliders();
    initializeTabs();

});


/* ===========================================================
   TAB SWITCHING
   =========================================================== */

function initializeTabs() {

    switchTab("single");

}

function switchTab(mode) {

    const tabs = document.querySelectorAll(".tab");
    tabs.forEach(tab => tab.classList.remove("active"));

    if (mode === "single") {

        tabs[0].classList.add("active");

        document
            .getElementById("tab-single")
            .classList.add("active");

        document
            .getElementById("tab-batch")
            .classList.remove("active");

    } else {

        tabs[1].classList.add("active");

        document
            .getElementById("tab-single")
            .classList.remove("active");

        document
            .getElementById("tab-batch")
            .classList.add("active");
    }
}


/* ===========================================================
   SLIDER BACKGROUND
   =========================================================== */

function initializeSliders() {

    document
        .querySelectorAll('input[type="range"]')
        .forEach(slider => updateSlider(slider));

}

function updateSlider(slider) {

    const min = slider.min;
    const max = slider.max;
    const val = slider.value;

    const pct = ((val - min) / (max - min)) * 100;

    slider.style.background =
        `linear-gradient(to right,
        #3B82F6 0%,
        #3B82F6 ${pct}%,
        rgba(255,255,255,.12) ${pct}%,
        rgba(255,255,255,.12) 100%)`;
}


/* ===========================================================
   TOGGLE BUTTONS
   =========================================================== */

function setToggle(button, hiddenId) {

    const container = button.parentElement;

    container
        .querySelectorAll(".toggle-opt")
        .forEach(btn => btn.classList.remove("sel"));

    button.classList.add("sel");

    document.getElementById(hiddenId).value =
        button.dataset.val;

}


/* ===========================================================
   SINGLE PREDICTION
   =========================================================== */

async function runSinglePredict(event) {

    event.preventDefault();

    const button =
        document.getElementById("predict-btn");

    button.disabled = true;
    button.innerHTML = "Predicting...";

    try {

        const form =
            document.getElementById("single-form");

        const formData = new FormData(form);

        const payload = {

            CreditScore: Number(formData.get("CreditScore")),
            Geography: Number(formData.get("Geography")),
            Gender: Number(formData.get("Gender")),
            Age: Number(formData.get("Age")),
            Tenure: Number(formData.get("Tenure")),
            Balance: Number(formData.get("Balance")),
            NumOfProducts: Number(formData.get("NumOfProducts")),
            HasCrCard: Number(formData.get("HasCrCard")),
            IsActiveMember: Number(formData.get("IsActiveMember")),
            EstimatedSalary: Number(formData.get("EstimatedSalary"))

        };

        const response = await fetch("/api/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(payload)

        });

        const result = await response.json();

        if (!response.ok) {

            throw new Error(result.detail || "Prediction failed");

        }

        console.log(result);
        updatePredictionUI(result, payload);
    }

    catch (err) {

        alert(err.message);

        console.error(err);

    }

    finally {

        button.disabled = false;

        button.innerHTML = `
        <svg width="16" height="16"
             viewBox="0 0 24 24"
             fill="none"
             stroke="currentColor"
             stroke-width="2">
          <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
        </svg>
        Run Prediction`;

    }

}


/* ===========================================================
   PART 2
   Prediction UI
   Gauge Animation
   Risk Card
   Input Summary
   =========================================================== */


/* -----------------------------------------------------------
   Called from Part 1 after successful prediction
------------------------------------------------------------*/

function updatePredictionUI(result, inputData) {

    // Hide empty state
    document.getElementById("empty-state").style.display = "none";

    // Show gauge
    document.getElementById("gauge-display").style.display = "block";

    // Animate gauge
    updateGauge(result.churn_probability);

    // Risk card
    updateRiskCard(
        result.risk_tier,
        result.churn_probability,
        result.prediction
    );

    // Input summary
    updateInputSummary(inputData);

}


/* -----------------------------------------------------------
   Gauge Animation
------------------------------------------------------------*/

function updateGauge(probability) {

    probability = Number(probability);

    // Accept either 0-1 or 0-100
    if (probability <= 1)
        probability *= 100;

    probability = Math.max(0, Math.min(100, probability));

    //---------------------------------------
    // Number
    //---------------------------------------

    document.getElementById("gauge-number").innerHTML =
        probability.toFixed(1) + "%";

    //---------------------------------------
    // Arc
    //---------------------------------------

    const circumference = 408;

    const offset =
        circumference -
        (probability / 100) * circumference;

    document
        .getElementById("gauge-fill")
        .style.strokeDashoffset = offset;

    //---------------------------------------
    // Needle
    //---------------------------------------

    const angle =
        -90 + probability * 1.8;

    document
        .getElementById("gauge-needle")
        .style.transform =
        `rotate(${angle}deg)`;

}


/* -----------------------------------------------------------
   Risk Card
------------------------------------------------------------*/

function updateRiskCard(risk, probability, prediction) {

    const card =
        document.getElementById("risk-card");

    const icon =
        document.getElementById("risk-icon");

    const title =
        document.getElementById("risk-title");

    const sub =
        document.getElementById("risk-sub");

    card.style.display = "flex";

    card.classList.remove(
        "risk-low",
        "risk-medium",
        "risk-high"
    );

    //----------------------------------------
    // LOW
    //----------------------------------------

    if (risk === "Low") {

        card.classList.add("risk-low");

        icon.innerHTML = "🟢";

        title.innerHTML =
            "Low Churn Risk";

        sub.innerHTML =
            `${probability.toFixed(1)}% probability`;

    }

    //----------------------------------------
    // MEDIUM
    //----------------------------------------

    else if (risk === "Medium") {

        card.classList.add("risk-medium");

        icon.innerHTML = "🟡";

        title.innerHTML =
            "Medium Churn Risk";

        sub.innerHTML =
            `${probability.toFixed(1)}% probability`;

    }

    //----------------------------------------
    // HIGH
    //----------------------------------------

    else {

        card.classList.add("risk-high");

        icon.innerHTML = "🔴";

        title.innerHTML =
            "High Churn Risk";

        sub.innerHTML =
            `${probability.toFixed(1)}% probability`;

    }

}


/* -----------------------------------------------------------
   Input Summary Card
------------------------------------------------------------*/

function updateInputSummary(data) {

    const card =
        document.getElementById("input-summary-card");

    const body =
        document.getElementById("input-summary-content");

    card.style.display = "block";

    const geo = {

        0: "France 🇫🇷",
        1: "Germany 🇩🇪",
        2: "Spain 🇪🇸"

    };

    const gender = {

        0: "Female",
        1: "Male"

    };

    body.innerHTML = `

<div class="summary-grid">

<div class="summary-item">
<strong>Country</strong>
<span>${geo[data.Geography]}</span>
</div>

<div class="summary-item">
<strong>Gender</strong>
<span>${gender[data.Gender]}</span>
</div>

<div class="summary-item">
<strong>Age</strong>
<span>${data.Age}</span>
</div>

<div class="summary-item">
<strong>Credit Score</strong>
<span>${data.CreditScore}</span>
</div>

<div class="summary-item">
<strong>Balance</strong>
<span>€${Number(data.Balance).toLocaleString()}</span>
</div>

<div class="summary-item">
<strong>Salary</strong>
<span>€${Number(data.EstimatedSalary).toLocaleString()}</span>
</div>

<div class="summary-item">
<strong>Products</strong>
<span>${data.NumOfProducts}</span>
</div>

<div class="summary-item">
<strong>Tenure</strong>
<span>${data.Tenure} yrs</span>
</div>

<div class="summary-item">
<strong>Credit Card</strong>
<span>${data.HasCrCard ? "Yes" : "No"}</span>
</div>

<div class="summary-item">
<strong>Active</strong>
<span>${data.IsActiveMember ? "Yes" : "No"}</span>
</div>

`;

}



// =========================
// Tab Switching
// =========================
function switchTab(tab){

    document.querySelectorAll(".tab").forEach(t=>{
        t.classList.remove("active");
    });

    document.querySelectorAll(".form-section").forEach(s=>{
        s.classList.remove("active");
    });

    if(tab==="single"){
        document.querySelectorAll(".tab")[0].classList.add("active");
        document.getElementById("tab-single").classList.add("active");
    }else{
        document.querySelectorAll(".tab")[1].classList.add("active");
        document.getElementById("tab-batch").classList.add("active");
    }

}


// =========================
// Toggle Buttons
// =========================
function setToggle(btn,inputId){

    const parent=btn.parentElement;

    parent.querySelectorAll(".toggle-opt").forEach(b=>{
        b.classList.remove("sel");
    });

    btn.classList.add("sel");

    document.getElementById(inputId).value=btn.dataset.val;

}


// =========================
// Slider Background
// =========================
function updateSlider(slider){

    const min=slider.min;
    const max=slider.max;
    const value=slider.value;

    const percent=((value-min)/(max-min))*100;

    slider.style.background=
    `linear-gradient(to right,
        #3B82F6 0%,
        #3B82F6 ${percent}%,
        rgba(255,255,255,.12) ${percent}%,
        rgba(255,255,255,.12) 100%)`;

}


// initialize sliders
document.querySelectorAll("input[type=range]").forEach(updateSlider);


// =========================
// Single Prediction
// =========================
async function runSinglePredict(event){

    event.preventDefault();

    const btn=document.getElementById("predict-btn");

    btn.disabled=true;
    btn.innerHTML="Predicting...";

    const form=document.getElementById("single-form");

    const data=Object.fromEntries(new FormData(form));

    Object.keys(data).forEach(k=>{
        data[k]=Number(data[k]);
    });

    try{

        const response=await fetch("/api/predict",{

            method:"POST",

            headers:{
                "Content-Type":"application/json"
            },

            body:JSON.stringify(data)

        });

        const result=await response.json();

        if(!response.ok){

            alert(result.detail || "Prediction failed");

            btn.disabled=false;
            btn.innerHTML="Run Prediction";

            return;

        }

        updatePredictionUI(result,data);

    }

    catch(err){

        alert("Server connection failed.");

        console.error(err);

    }

    btn.disabled=false;

    btn.innerHTML="Run Prediction";

}



// =========================
// Update Right Panel
// =========================
function updatePredictionUI(result,input){

    document.getElementById("empty-state").style.display="none";
    document.getElementById("gauge-display").style.display="block";

    const prob=result.churn_probability;

    document.getElementById("gauge-number").innerHTML=prob.toFixed(1)+"%";

    const circumference=408;

    const offset=circumference-(prob/100)*circumference;

    document.getElementById("gauge-fill")
        .style.strokeDashoffset=offset;

    const angle=(prob/100)*180-90;

    document.getElementById("gauge-needle")
        .style.transform=`rotate(${angle}deg)`;


    const risk=document.getElementById("risk-card");
    risk.style.display="flex";

    document.getElementById("risk-title").innerHTML=
        result.risk_tier+" Risk";

    document.getElementById("risk-sub").innerHTML=
        result.prediction===1 ?
        "Customer likely to churn":
        "Customer likely to stay";


    let icon="🟢";

    if(result.risk_tier==="Medium")
        icon="🟡";

    if(result.risk_tier==="High")
        icon="🔴";

    document.getElementById("risk-icon").innerHTML=icon;


    const summary=document.getElementById("input-summary-card");

    summary.style.display="block";

    document.getElementById("input-summary-content").innerHTML=`
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:.9rem">

        <div><strong>Age</strong><br>${input.Age}</div>

        <div><strong>Credit Score</strong><br>${input.CreditScore}</div>

        <div><strong>Balance</strong><br>€${Number(input.Balance).toLocaleString()}</div>

        <div><strong>Salary</strong><br>€${Number(input.EstimatedSalary).toLocaleString()}</div>

        <div><strong>Products</strong><br>${input.NumOfProducts}</div>

        <div><strong>Tenure</strong><br>${input.Tenure} years</div>

        </div>
    `;

}



// =========================
// Batch Upload
// =========================
let selectedFile=null;

function handleFileSelect(input){

    if(input.files.length===0) return;

    selectedFile=input.files[0];

    document.getElementById("file-name-display").innerHTML=
        selectedFile.name;

    document.getElementById("batch-btn").disabled=false;

}


function handleDrop(event){

    event.preventDefault();

    document.getElementById("dropzone")
        .classList.remove("over");

    selectedFile=event.dataTransfer.files[0];

    document.getElementById("batch-file").files=
        event.dataTransfer.files;

    handleFileSelect(document.getElementById("batch-file"));

}



async function runBatchPredict(){

    if(!selectedFile) return;

    const fd=new FormData();

    fd.append("file",selectedFile);

    const btn=document.getElementById("batch-btn");

    btn.disabled=true;

    btn.innerHTML="Uploading...";

    try{

        const response=await fetch("/predict_batch",{

            method:"POST",

            body:fd

        });

        const result=await response.json();

        if(!response.ok){

            alert(result.detail);

            btn.disabled=false;

            btn.innerHTML="Predict Batch";

            return;

        }

        document.getElementById("batch-summary").style.display="block";

        document.getElementById("batch-summary-content").innerHTML=`
            <strong>${result.num_predictions}</strong>
            customers processed successfully.
        `;

        const dl=document.getElementById("batch-download");

        dl.style.display="inline-flex";

        dl.href=result.download_url;

    }

    catch(err){

        alert("Batch prediction failed.");

        console.error(err);

    }

    btn.disabled=false;

    btn.innerHTML="Predict Batch";

}