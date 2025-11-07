export default function ProgressBar({value}:{value:number}){
  return <div className="progress" aria-hidden="true"><span style={{width:`${value}%`}}/></div>;
}
