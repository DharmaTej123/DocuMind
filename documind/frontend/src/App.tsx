import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AppShell } from "@/components/AppShell";
import { RequireAuth } from "@/components/RequireAuth";
import { LoginPage } from "@/pages/LoginPage";
import { RegisterPage } from "@/pages/RegisterPage";
import { AskPage } from "@/pages/AskPage";
import { DocumentsPage } from "@/pages/DocumentsPage";
import { HistoryPage } from "@/pages/HistoryPage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route
          element={
            <RequireAuth>
              <AppShell />
            </RequireAuth>
          }
        >
          <Route path="/" element={<AskPage />} />
          <Route path="/documents" element={<DocumentsPage />} />
          <Route path="/history" element={<HistoryPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
