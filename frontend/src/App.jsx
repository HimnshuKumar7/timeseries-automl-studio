import React, { useEffect, useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Wizard from './components/Wizard';
import LoginPage from './pages/LoginPage';
import ErrorBoundary from './components/ErrorBoundary';

function useAuthStatus() {
  const [status, setStatus] = useState('checking'); // 'checking' | 'authed' | 'anon'

  useEffect(() => {
    let cancelled = false;
    fetch('/api/auth/me', { credentials: 'include' })
      .then((res) => {
        if (!cancelled) setStatus(res.ok ? 'authed' : 'anon');
      })
      .catch(() => {
        if (!cancelled) setStatus('anon');
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return status;
}

function LoadingScreen() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-950 text-gray-500 dark:text-gray-400 text-sm">
      <div className="flex items-center space-x-3">
        <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
        <span>Loading AutoML Studio...</span>
      </div>
    </div>
  );
}

function RequireAuth({ children }) {
  const status = useAuthStatus();
  if (status === 'checking') return <LoadingScreen />;
  if (status === 'anon') return <Navigate to="/login" replace />;
  return <>{children}</>;
}

function RedirectIfAuthed({ children }) {
  const status = useAuthStatus();
  if (status === 'checking') return <LoadingScreen />;
  if (status === 'authed') return <Navigate to="/" replace />;
  return <>{children}</>;
}

export default function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <Routes>
          <Route
            path="/login"
            element={
              <RedirectIfAuthed>
                <LoginPage />
              </RedirectIfAuthed>
            }
          />
          <Route
            path="/"
            element={
              <RequireAuth>
                <Wizard />
              </RequireAuth>
            }
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </ErrorBoundary>
  );
}
