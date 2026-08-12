import { useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { api } from "@/lib/api";

export function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await api.login(email, password);
      localStorage.setItem("documind_token", res.access_token);
      navigate("/");
    } catch {
      setError("Couldn't sign you in. Check your email and password.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-ink px-4">
      <Card className="w-full max-w-sm p-6">
        <p className="font-display text-xl font-semibold text-ink">DocuMind</p>
        <p className="mt-1 text-sm text-muted">
          Sign in to your workspace
        </p>

        <form onSubmit={handleSubmit} className="mt-5 flex flex-col gap-3">
          <div>
            <label className="mb-1 block font-mono text-xs text-muted">email</label>
            <Input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@company.com"
            />
          </div>
          <div>
            <label className="mb-1 block font-mono text-xs text-muted">password</label>
            <Input
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
            />
          </div>

          {error && <p className="text-sm text-danger">{error}</p>}

          <Button type="submit" variant="signal" disabled={loading} className="mt-2">
            {loading ? "Signing in…" : "Sign in"}
          </Button>
        </form>

        <p className="mt-4 text-sm text-muted">
          No account?{" "}
          <Link to="/register" className="text-signal hover:underline">
            Create one
          </Link>
        </p>
      </Card>
    </div>
  );
}
