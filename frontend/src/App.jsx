import Dashboard from "./Dashboard";
import { useMemo } from "react";

function App() {

  const dots = useMemo(() => {

    const rows = 10
    const cols = 14
    const arr = []

    for(let r=0;r<rows;r++){
      for(let c=0;c<cols;c++){

        const size = Math.random()*3 + 2

        const left = (c/(cols-1))*100
        const top = (r/(rows-1))*100

        const duration = Math.random()*20 + 15
        const delay = Math.random()*20

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

  return (

    <div className="dark">

      {/* background */}
      <div className="energyBackground"></div>

      {/* floating particles */}
      <div className="energyDots">
        {dots}
      </div>

      {/* dashboard */}
      <Dashboard />

    </div>

  )

}

export default App