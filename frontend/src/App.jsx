import Dashboard from "./Dashboard"
import { useState, useEffect, useMemo } from "react"

function App(){

const [stage,setStage] = useState(0)
const [progress,setProgress] = useState(0)
const [stepIndex,setStepIndex] = useState(0)
const [fadeEnter,setFadeEnter] = useState(false)

const steps = [
"Initializing Energy Forecast System...",
"Loading Transformer Model...",
"Preparing Dataset Interface...",
"Launching Dashboard..."
]

useEffect(() => {

const favicon = document.getElementById("favicon")

const handleVisibility = () => {

if (document.hidden) {
favicon.href = "/ghost.png"
} else {
favicon.href = "/energy-icon.png"
}

}

document.addEventListener("visibilitychange", handleVisibility)

return () => {
document.removeEventListener("visibilitychange", handleVisibility)
}

}, [])
useEffect(()=>{

const interval = setInterval(()=>{

setProgress(p=>{

const next = p + 1.5

// change message every 25%
if(next >= 25 && stepIndex === 0) setStepIndex(1)
if(next >= 50 && stepIndex === 1) setStepIndex(2)
if(next >= 75 && stepIndex === 2) setStepIndex(3)

if(next >= 100){
clearInterval(interval)
setTimeout(()=>setStage(1),300)
return 100
}

return next

})

},40)

return ()=>clearInterval(interval)

},[stepIndex])

const dots = useMemo(()=>{

const rows=10
const cols=14
const arr=[]

for(let r=0;r<rows;r++){
for(let c=0;c<cols;c++){

const size=Math.random()*3+2
const left=(c/(cols-1))*100
const top=(r/(rows-1))*100
const duration=Math.random()*20+15
const delay=Math.random()*20

arr.push(

<div
key={r+"-"+c}
className="energyDot"
style={{
left:left+"%",
top:top+"%",
width:size+"px",
height:size+"px",
animationDuration:duration+"s",
animationDelay:delay+"s"
}}
/>

)

}
}

return arr

},[])

return(

<div className="dark">

<div className="energyBackground"></div>

<div className="energyDots">{dots}</div>

{/* BOOT SCREEN */}

{stage===0 && (

<div className="fixed inset-0 flex flex-col items-center justify-center text-center z-50 bootFade">
<img
src="/loading-car.gif"
className="w-24 mb-8 animate-pulse"
/>

<h1 className="text-3xl text-white mb-6 tracking-wide">

AI Energy Forecast System

</h1>

<div className="w-72 bg-gray-700 rounded-full h-2 mb-3 overflow-hidden">

<div
className="bg-indigo-500 h-2 transition-[width] duration-200 ease-out"
style={{width:progress+"%"}}
/>

</div>

    <p className="text-gray-400 text-sm transition-opacity duration-300">
        {steps[stepIndex]}
            </p>

</div>

)}

{/* ENTER SCREEN */}

{stage===1 && (

<div className={`fixed inset-0 flex flex-col items-center justify-center text-center z-50 transition-all duration-500 ${fadeEnter ? "opacity-0 scale-95" : "opacity-100 scale-100"}`}>
<img
src="/light-icon.gif"
className="w-24 iconGlow "
/>

<h1 className="text-4xl font-semibold text-white mb-4">
AI Energy Load Forecast
</h1>

<p className="text-gray-400 max-w-lg mb-8">
Predict electricity demand using Transformer-based
deep learning models with explainable AI insights.
</p>

<button
onClick={()=>{
setFadeEnter(true)

setTimeout(()=>{
setStage(2)
},500)
}}
className="enterButton"
>
Enter Dashboard
</button>

</div>

)}

{/* DASHBOARD */}

{stage===2 && (
<div className="dashboardFade">
<Dashboard/>
</div>
)}
</div>

)

}

export default App