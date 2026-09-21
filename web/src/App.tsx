import { Rail } from "./components/Chrome";
import { Engine } from "./sections/Engine";
import { Harness } from "./sections/Harness";
import { Opening } from "./sections/Opening";
import { Pivot } from "./sections/Pivot";

export default function App() {
  return (
    <div className="page">
      <Rail />
      <main className="main">
        <div className="wrap">
          <Opening />
          <Engine />
          <Pivot />
          <Harness />
        </div>
      </main>
    </div>
  );
}
