import { Navbar } from '@/components/Navbar';
import { DocumentList } from '@/components/DocumentList';

export default function HomePage() {
  return (
    <div className="min-h-screen bg-slate-950 flex flex-col">
      <Navbar />
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <DocumentList />
      </main>
      <footer className="border-t border-slate-900 py-6 text-center text-xs text-slate-400">
        Ajaia LLC • Full Stack Product Engineer Assignment • Built by Vaibhavi Diwakar
      </footer>
    </div>
  );
}
