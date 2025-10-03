import { Outlet } from "react-router-dom";
import { Navigation } from "./Navigation";
import ChatWidget from "./ChatWidget";

export function Layout() {
  return (
    <div className="min-h-screen">
      <Navigation />
      <main>
        <Outlet />
      </main>
      {/* Global ChatWidget available on all pages */}
      <ChatWidget />
    </div>
  );
}