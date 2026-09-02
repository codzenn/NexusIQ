import { checkHealth } from '@/lib/api';

export default async function Home() {
    const health = await checkHealth();

    return (
        <main className="min-h-screen p-10">
            <div className="mx-auto max-w-5xl">
                <h1 className="text-4xl font-bold">NexusIQ</h1>

                <p className="mt-3 text-muted-foreground">
                    Enterprise Agentic Intelligence Platform
                </p>

                <div className="mt-10 rounded-xl border p-6">
                    <h2 className="text-xl font-semibold">System Status</h2>

                    <p className="mt-3">
                        API:{' '}
                        <span className="font-medium">
                            {health.status === 'ok' ? 'Connected' : 'Unavailable'}
                        </span>
                    </p>
                </div>
            </div>
        </main>
    );
}
