import { useState, useMemo } from "react";
import axios from "axios";
import { Disclosure } from "@headlessui/react";

export default function Dashboard() {


// ====================== STATE VARIABLES ======================

// uploaded CSV
const [file,setFile]=useState(null)

// forecast or evaluate
const [mode,setMode]=useState("forecast")

// API result
const [data,setData]=useState(null)

// loading indicator
const [loading,setLoading]=useState(false)

// active tab (24h / 7d / 30d)
const [activeTab,setActiveTab]=useState("24h")

// search filter
const [search,setSearch]=useState("")

// sorting
const [sortKey,setSortKey]=useState(null)
const [sortState,setSortState]=useState(null) // asc desc null



// ====================== RUN MODEL ======================

const runModel = async ()=>{

if(!file) return alert("Upload CSV file")

const formData = new FormData()
formData.append("file",file)
formData.append("mode",mode)

try{

setLoading(true)

const res = await axios.post(
"https://energy-forecast-api-sfrz.onrender.com/run_model",
formData
)

setData(res.data)

}catch(err){

console.error(err)
if(err.response && err.response.data && err.response.data.detail){
alert(err.response.data.detail)
}
else{
alert("Server Error")
}

}finally{

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


// ====================== GRAPH ======================

const getGraph = ()=>{

if(!data) return ""

return "https://energy-forecast-api-sfrz.onrender.com"+data.graphs?.[activeTab]

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


// ====================== UI ======================

return(

<div className="min-h-screen bg-gray-900 text-gray-200 p-6">


{/* ================= TITLE ================= */}

<h1 className="text-2xl font-bold mb-6 text-center">
⚡ Energy Load Forecast Dashboard
</h1>



{/* ================= UPLOAD PANEL ================= */}

<div className="bg-gray-800 p-4 rounded-lg flex gap-3 flex-wrap justify-center">


<input
type="file"
accept=".csv"
onChange={(e)=>setFile(e.target.files[0])}
className="bg-gray-700 text-sm p-2 rounded"
/>


<select
value={mode}
onChange={(e)=>setMode(e.target.value)}
className="bg-gray-700 text-sm p-2 rounded"
>
<option value="forecast">Forecast</option>
<option value="evaluate">Evaluate</option>
</select>


<button
onClick={runModel}
className="bg-indigo-600 px-4 py-2 text-sm rounded hover:bg-indigo-700 transition"
>
{loading?"Running...":"Run Model"}
</button>

</div>
 


{/* ================= EVALUATION METRICS ================= */}

{mode==="evaluate" && data?.evaluation &&(

<div className="grid grid-cols-3 gap-4 mt-6 text-sm">

{["24H","7D","30D"].map(p=>(

<div key={p} className="bg-gray-800 p-4 rounded-lg text-center hover:bg-gray-700 transition-colors">

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

<div className="bg-gray-800 p-3 mt-4 rounded-lg text-center text-sm w-[60%] mx-auto">

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
onClick={()=>setActiveTab(tab)}
className={`px-4 py-1 rounded ${
activeTab===tab ? "bg-indigo-600 hover:bg-indigo-700 transition" : "bg-gray-700 hover:bg-gray-600 transition"
}`}
>

{tab.toUpperCase()}

</button>

))}

</div>



{/* ================= FORECAST TABLE ================= */}

<div className="bg-gray-800 p-4 mt-4 rounded-lg w-[80%] mx-auto">

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
Export CSV
</button>

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
Load (MW)
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
Load (MW)
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
<td className="py-2 text-center">{row.load_mw.toFixed(4)}</td>
</>

):( 

<>
<td className="py-2 text-center">{row.date}</td>
<td className="py-2 text-center">{row.day}</td>
<td className="py-2 text-center">{row.load_mw.toFixed(4)}</td>
</>

)}

</tr>

)

})}

</tbody>

</table>

</div>

</div>



{/* ================= GRAPH ================= */}

<div className="bg-gray-800 p-4 mt-4 rounded-lg ">

<h2 className="text-lg mb-2 text-center">
Forecast Graph
</h2>

<iframe
src={getGraph()}
className="w-full h-[400px] hover:bg-pink-300 rounded-lg transition-colors"
/>

</div>



{/* ================= EXPLANATIONS ================= */}

<div className="mt-4 space-y-3">

{["trend","academic","attention","features"].map(section=>(

<Disclosure key={section}>

<div className="bg-gray-800 rounded-lg hover:bg-gray-700 transition-colors">

<Disclosure.Button className="w-full px-4 py-2 text-left text-sm font-semibold">
{section.toUpperCase()}
</Disclosure.Button>

<Disclosure.Panel className="px-4 pb-3 text-xs whitespace-pre-line text-gray-400">
{getExplanation(section)}
</Disclosure.Panel>

</div>

</Disclosure>

))}

</div>

</>

)}

</div>

)

}