import { useState, useMemo } from "react";
import axios from "axios";
import { Disclosure, Transition } from "@headlessui/react";
import About from "./About"
import Contact from "./Contact"
import FeatureInfluenceChart from "./components/FeatureInfluenceChart"

export default function Dashboard() {

// ====================== STATE VARIABLES ======================
const rows = 10
const cols = 14
const API_URL = "https://energy-forecast-api-sfrz.onrender.com"
const dots = []

for(let r=0;r<rows;r++){
  for(let c=0;c<cols;c++){

    const size = Math.random()*3 + 2

    const left = (c/(cols-1))*100 + (Math.random()*2 - 1)
    const top  = (r/(rows-1))*100 + (Math.random()*2 - 1)

    const duration = Math.random()*20 + 15
    const delay = Math.random()*20

    dots.push(

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

// uploaded CSV
const [file,setFile]=useState(null)

// forecast or evaluate
const [mode,setMode]=useState("forecast")

// API result
const [data,setData]=useState(null)

// loading indicator
const [loading,setLoading]=useState(false)

// progress for liquid loader
const [progress,setProgress]=useState(0)

// active tab (24h / 7d / 30d)
const [activeTab,setActiveTab]=useState("24h")

// search filter
const [search,setSearch]=useState("")

// sorting
const [sortKey,setSortKey]=useState(null)
const [sortState,setSortState]=useState(null) // asc desc null

const [fade,setFade] = useState("fadeIn")

const [showAbout,setShowAbout] = useState(false)
const [showContact,setShowContact] = useState(false)

// ====================== RUN MODEL ======================

const runModel = async ()=>{

if(!file) return alert("Upload CSV file")

const formData = new FormData()
formData.append("file",file)
formData.append("mode",mode)

try{

setLoading(true)
setProgress(5)

// simulated progress animation
const progressInterval = setInterval(()=>{
setProgress(p=>{
if(p < 90) return p + 2
return p
})
},800)

const res = await axios.post(
`${API_URL}/run_model`,
formData
)

clearInterval(progressInterval)

setProgress(100)

setTimeout(()=>{
setData(res.data)
setLoading(false)
setProgress(0)
},800)

}catch(err){

console.error(err)
if(err.response && err.response.data && err.response.data.detail){
alert(err.response.data.detail)
}
else{
alert("Server Error")
}

setLoading(false)

}

}

// ====================== GET FORECAST ======================

const getForecast = ()=>{

if(!data) return []

if(activeTab==="24h") return data["24h_forecast"]||[]
if(activeTab==="7d") return data["7d_forecast"]||[]

return data["30d_forecast"]||[]

}

// ====================== SORT FUNCTION ======================

const handleSort=(key)=>{

if(sortKey!==key){
setSortKey(key)
setSortState("asc")
return
}

if(sortState==="asc"){
setSortState("desc")
}
else if(sortState==="desc"){
setSortKey(null)
setSortState(null)
}
else{
setSortState("asc")
}

}

// ====================== EXPORT CSV ======================

const exportCSV = ()=>{

const rows=getForecast()

let csv="Time/Date,Load(MW)\n"

rows.forEach(r=>{
const t=r.time||r.date
csv+=`${t},${r.load_mw}\n`
})

const blob=new Blob([csv],{type:"text/csv"})
const url=URL.createObjectURL(blob)

const a=document.createElement("a")
a.href=url
a.download="forecast.csv"
a.click()

}
// ====================== DOWNLOAD GRAPHS ====================

const downloadGraph = () => {

const graphUrl = getGraph()

if(!graphUrl) return

const a = document.createElement("a")
a.href = graphUrl
a.download = `forecast_${activeTab}.html`
a.target = "_blank"

a.click()

}

// ====================== FILTER + SORT ======================

const processedForecast = useMemo(()=>{

let rows=[...getForecast()]

if(search){

rows=rows.filter(r=>
(r.time||r.date)
.toLowerCase()
.includes(search.toLowerCase())
)

}

if(sortKey && sortState){

rows.sort((a,b)=>{

const A=a[sortKey]
const B=b[sortKey]

if(sortState==="asc") return A-B
if(sortState==="desc") return B-A
return 0

})

}

return rows

},[data,activeTab,search,sortKey,sortState])

const maxLoad = Math.max(...processedForecast.map(r => r.load_mw))
const minLoad = Math.min(...processedForecast.map(r => r.load_mw))

// ====================== GRAPH ======================

const getGraph = ()=>{

if(!data) return ""

return API_URL + data.graphs?.[activeTab]
}

// ====================== EXPLANATIONS ======================

const getExplanation=(key)=>{

const horizon=
activeTab==="24h"
?"24H"
:activeTab==="7d"
?"7D"
:"30D"

return data?.explanations?.[horizon]?.[key]

}

{/* ================= FEATURE INFLUENCE ================= */}

{data?.feature_influence && (

<div className={`bg-gray-800/50 backdrop-blur-md p-4 mt-4 rounded-lg fadeContent ${fade}`}>

<h2 className="text-lg mb-3 text-center">
Feature Influence Analysis
</h2>

<FeatureInfluenceChart features={data.feature_influence}/>

</div>

)}

// ====================== LIGHTNING EFFECT ======================

const createLightning = (e)=>{

const canvas = document.createElement("canvas")
canvas.className="lightningCanvas"

canvas.width = window.innerWidth
canvas.height = window.innerHeight

document.body.appendChild(canvas)

const ctx = canvas.getContext("2d")

const startX = e.clientX
const startY = 0
const endY = e.clientY

function drawBolt(x,y,maxY,branch){

ctx.beginPath()
ctx.moveTo(x,y)

let currentX = x
let currentY = y

while(currentY < maxY){

currentX += (Math.random()-0.5)*30
currentY += Math.random()*25

ctx.lineTo(currentX,currentY)

if(branch && Math.random() > 0.85){

drawBolt(currentX,currentY,currentY+40,false)

}

}

ctx.stroke()

}

ctx.strokeStyle = "rgba(200,220,255,0.9)"
ctx.lineWidth = 2
ctx.shadowBlur = 20
ctx.shadowColor = "#818cf8"

drawBolt(startX,startY,endY,true)



/* screen flash */

const flash = document.createElement("div")
flash.className="lightningFlash"
document.body.appendChild(flash)



/* ripple energy */

const ripple = document.createElement("div")
ripple.className="energyRipple"

ripple.style.left = e.clientX+"px"
ripple.style.top = e.clientY+"px"

document.body.appendChild(ripple)



/* optional thunder sound */

// new Audio("/thunder.mp3").play()



setTimeout(()=>{
canvas.remove()
flash.remove()
ripple.remove()
},150)

}
// ====================== UI ======================

return(

<div className="min-h-screen flex flex-col text-gray-200 pt-16 px-6" 
    onClick={createLightning}>

<div className="flex-grow">
{/* ================= LOADER ================= */}

{loading && (

<div className="fixed inset-0 flex items-center justify-center bg-black/70 z-50">

<div className="flex flex-col items-center">

<div className="loader-glow">

<img
src="/lightning.png"
className="w-22 h-22 object-contain loader-color"
/>

</div>

<p className="mt-4 text-sm text-gray-300">
Running Energy Forecast Model...
</p>

<p className="text-sm text-gray-400 mt-1 text-center max-w-md">
<span className="text-red-400">Server Error</span> occurs at first time <span className="text-yellow-400">don't worry</span> - First request may take ~ 1-2 minutes to wake up server
</p>

</div>

</div>

)}

{/* ================= TITLE ================= */}

<h1 className="text-3xl font-semibold tracking-wide mb-6 text-center flex items-center justify-center gap-3">

<img
src="/energy.gif"
className="w-18 energy-iconColor"
/>
Energy Load Forecast Dashboard

</h1>

{/* ================= UPLOAD PANEL ================= */}

<div className="p-4 rounded-xl flex gap-3 flex-wrap justify-center ">
<input
type="file"
accept=".csv"
onChange={(e)=>setFile(e.target.files[0])}
className="bg-gray-700/45 backdrop-blur-md text-sm p-2 rounded"
/>

<select
value={mode}
onChange={(e)=>setMode(e.target.value)}
className="bg-gray-700/45 backdrop-blur-md text-sm p-2 rounded"

>

<option value="forecast">Forecast</option>
<option value="evaluate">Evaluate</option>
</select>

<button
onClick={runModel}
className="bg-indigo-600 px-4 py-2 text-sm rounded hover:bg-indigo-700 transition"

>

{loading?"Running...":"Run Model"} </button>

</div>

{/* ================= EVALUATION METRICS ================= */}

{mode==="evaluate" && data?.evaluation &&(

<div className="grid grid-cols-3 gap-4 mt-6 text-sm">

{["24H","7D","30D"].map(p=>(

<div key={p} className="bg-gray-900/50 backdrop-blur-md p-4 rounded-lg text-center hover:bg-gray-700 transition-colors">

<h3 className="font-semibold mb-1">{p}</h3>

<p>MAE: {data.evaluation[`${p}_MAE_MW`]?.toFixed(2)} MW</p>
<p className="text-red-400 font-semibold">Error: {data.evaluation[`${p}_Error_%`]?.toFixed(2)}%</p>

</div>

))}

</div>

)}

{/* SHOW RESULTS ONLY AFTER RUN MODEL */}

{data && (

<>
{/* ================= LAST AVAILABLE TIME ================= */}

<div className="p-3 mt-4 rounded-xl text-center text-sm w-[60%] mx-auto ">

<span className="text-gray-400">Last Available Data Time: </span>

<span className="text-white font-semibold">
{data.last_available_time}
</span>

</div>

{/* ================= FORECAST TABS ================= */}

<div className="flex gap-2 mt-6 text-sm justify-center">

{["24h","7d","30d"].map(tab=>(

<button
key={tab}
onClick={()=>{
setFade("fadeOut")

setTimeout(()=>{
setActiveTab(tab)
setFade("fadeIn")
},200)

}}   
className={`px-4 py-1 rounded ${
activeTab===tab ? "bg-indigo-600 hover:bg-indigo-700 transition" : "bg-gray-700 hover:bg-gray-600 transition"
}`}

>

{tab.toUpperCase()}

</button>

))}

</div>

{/* ================= FORECAST TABLE ================= */}

<div className={`bg-gray-800/50 backdrop-blur-md p-4 mt-4 rounded-lg w-[80%] mx-auto fadeContent ${fade}`}>

<h2 className="text-lg mb-3 text-center">
Forecast Values ({activeTab.toUpperCase()})
</h2>

<div className="flex justify-between mb-3">

<input
type="text"
placeholder="Search..."
value={search}
onChange={(e)=>setSearch(e.target.value)}
className="bg-gray-700 text-xs px-2 py-1 rounded hover:bg-gray-600 transition"
/>

<button
onClick={exportCSV}
className="bg-indigo-600 px-3 py-1 text-xs rounded hover:bg-indigo-700 transition"

>

Export CSV </button>

</div>

<div className="max-h-[320px] overflow-y-auto ">

<table className="w-[80%] mx-auto text-xs table-fixed ">

<thead>

<tr className="border-b border-white-600 text-gray-300">

{activeTab==="24h" ? (
<>

<th className="py-2 text-center  ">Time</th>

<th
onClick={()=>handleSort("load_mw")}
className="py-2 text-center cursor-pointer hover:text-gray-400 transition"
>
Load <span className="mono">MW</span>
{sortState==="asc" && " ▲"}
{sortState==="desc" && " ▼"}
</th>
</>
) : (
<>
<th className="py-2 text-center">Date</th>
<th className="py-2 text-center">Day</th>

<th
onClick={()=>handleSort("load_mw")}
className="py-2 text-center cursor-pointer hover:text-indigo-400 transition"
>
Load <span className="mono">MW</span>
{sortState==="asc" && " ▲"}
{sortState==="desc" && " ▼"}
</th>
</>
)}

</tr>

</thead>

<tbody>

{processedForecast.map((row,i)=>{

return(

<tr
key={i}
className="border-b border-gray-700 hover:bg-gray-700 cursor-pointer transition"
>
{activeTab==="24h" ? (

<>

<td className="py-2 text-center">{row.time}</td>
<td
className={`py-2 text-center font-semibold
${row.load_mw === maxLoad ? "text-red-400" : ""}
${row.load_mw === minLoad ? "text-green-400" : ""}
`}
>
{row.load_mw.toFixed(4)}
</td></>

):(

<>

<td className="py-2 text-center">{row.date}</td>
<td className="py-2 text-center">{row.day}</td>
<td
className={`py-2 text-center font-semibold
${row.load_mw === maxLoad ? "text-red-400" : ""}
${row.load_mw === minLoad ? "text-green-400" : ""}
`}
>
{row.load_mw.toFixed(4)}
</td></>

)}

</tr>

)

})}

</tbody>

</table>

</div>

</div>

{/* ================= GRAPH ================= */}

<div className={`bg-gray-800/50 backdrop-blur-md p-4 mt-4 rounded-lg fadeContent ${fade}`}>
<div className="flex justify-between items-center mb-2">

<h2 className="text-lg">
Forecast Graph
</h2>

<button
onClick={downloadGraph}
className="bg-indigo-600 px-3 py-1 text-xs rounded hover:bg-indigo-700 transition"
>
Download Graph
</button>

</div>
<iframe
src={getGraph()}
className="w-full h-[450px] hover:bg-indigo-500 rounded-lg transition-colors"
/>

</div>

{/* ================= EXPLANATIONS ================= */}

<div className={`mt-4 space-y-3 fadeContent ${fade}`} key={activeTab}>

{["trend","academic","features"].map(section=>(
<Disclosure key={section}>

<div className="bg-gray-800/50 backdrop-blur-md rounded-lg hover:bg-gray-700 transition-colors overflow-hidden">
<Disclosure.Button className="w-full px-4 py-2 text-left text-sm font-semibold transition-all duration-300 hover:text-indigo-400 hover:pl-5">{section.toUpperCase()}

</Disclosure.Button>

<Transition
enter="transition-all duration-500 ease-out"
enterFrom="opacity-0 -translate-y-3 scale-95"
enterTo="opacity-100 translate-y-0 scale-100"
leave="transition-all duration-300 ease-in"
leaveFrom="opacity-100 translate-y-0 scale-100"
leaveTo="opacity-0 -translate-y-2 scale-95"
>
<Disclosure.Panel className="px-5 pb-4 text-sm leading-relaxed text-gray-300 font-light whitespace-pre-line">

{getExplanation(section)}

{section==="features" && data?.feature_influence?.length>0 && (

<div className="mt-4">

<FeatureInfluenceChart features={data.feature_influence}/>

</div>

)}

</Disclosure.Panel>
</Transition>

</div>

</Disclosure>

))}

</div>

</>

)}
</div>

{/* ABOUT PANEL */}
{showAbout && (
<div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex justify-end">

<div className="w-[420px] h-full bg-gray-900 border-l border-gray-700 animate-slidePanel relative">

<button
onClick={()=>setShowAbout(false)}
className="absolute top-5 right-6 text-gray-400 hover:text-white"
>
✕
</button>

<About/>

</div>

</div>
)}

{/* CONTACT PANEL */}
{showContact && (
<div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex justify-end">

<div className="w-[380px] h-full bg-gray-900 border-l border-gray-700 animate-slidePanel relative">

<button
onClick={()=>setShowContact(false)}
className="absolute top-5 right-6 text-gray-400 hover:text-white"
>
✕
</button>

<Contact/>

</div>

</div>
)}

{/* ================= FOOTER ================= */}
<footer className="w-full border-t border-gray-700 py-3 flex justify-center items-center gap-10 text-sm text-gray-400">

<button
onClick={()=>setShowAbout(true)}
className="hover:text-white transition"
>
About
</button>

<button
onClick={()=>setShowContact(true)}
className="hover:text-white transition"
>
Contact
</button>

</footer>
</div>

)

}

