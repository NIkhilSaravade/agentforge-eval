import { Rail } from "./components/Chrome";
import { Engine } from "./sections/Engine";
import { Opening } from "./sections/Opening";

export default function App() {
  return (
    <div className="page">
      <Rail />
      <main className="main">
        <div className="wrap">
          <Opening />
          <Engine />
        </div>
      </main>
    </div>
  );
}
