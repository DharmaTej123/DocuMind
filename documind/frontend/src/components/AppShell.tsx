import { Outlet, useNavigate } from "react-router-dom";
import { Sidebar } from "@/components/Sidebar";

export function AppShell() {
  const navigate = useNavigate();

  function handleLogout() {
    localStorage.removeItem("documind_token");
    navigate("/login");
  }

  return (
    <div className="flex">
      <Sidebar onLogout={handleLogout} />
      <main className="min-h-screen flex-1 bg-paper">
        <Outlet />
      </main>
    </div>
  );
}
