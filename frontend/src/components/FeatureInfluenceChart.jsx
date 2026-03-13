export default function FeatureInfluenceChart({features}){

if(!features || features.length===0) return null

const maxScore = Math.max(...features.map(f=>f.score))

return(

<div className="space-y-3 max-w-xl">

{features.map((f,i)=>{

const width = (f.score/maxScore)*100

return(

<div key={i} className="flex items-center gap-3">

{/* Feature name */}
<div className="w-36 text-xs text-gray-300 truncate">
{f.feature}
</div>

{/* bar */}
<div className="flex-1 max-w-md bg-gray-700 rounded h-2 overflow-hidden">

<div
className="h-full bg-gradient-to-r from-indigo-800 to-cyan-400 transition-all duration-500"
style={{width:`${width}%`}}
/>

</div>

</div>

)

})}

</div>

)

}   