import { Rail, ThemeWatcher } from "./components/Chrome";
import { Close } from "./sections/Close";
import { Engine } from "./sections/Engine";
import { Harness } from "./sections/Harness";
import { Opening } from "./sections/Opening";
import { Pivot } from "./sections/Pivot";
import { Results } from "./sections/Results";

export default function App() {
  return (
    <div className="page">
      <ThemeWatcher />
      <Rail />
      <main className="main">
        <div className="wrap">
          <Opening />
          <Engine />
          <Pivot />
          <Harness />
          <Results />
          <Close />
        </div>
      </main>
    </div>
  );
}
