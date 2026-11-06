import numpy as np
import plotly.graph_objects as go

# La sesión debe haberse cargado en la celda anterior.
circuit_info = session.get_circuit_info()
lap_ver = session.laps.pick_drivers('VER').pick_fastest()
lap_lec = session.laps.pick_drivers('LEC').pick_fastest()


def posiciones_vuelta(lap):
    """Obtiene posiciones y tiempos relativos a la salida de la vuelta."""
    if lap is None or lap['LapStartTime'] is None or lap['Time'] is None:
        raise ValueError('No se encontró una vuelta rápida con tiempos válidos.')
    # Las muestras exteriores permiten interpolar las posiciones exactas en la meta.
    pos = lap.get_pos_data(pad=1, pad_side='both').dropna(subset=['SessionTime', 'X', 'Y'])
    pos = pos.sort_values('SessionTime').drop_duplicates('SessionTime')
    tiempos = (pos['SessionTime'] - lap['LapStartTime']).dt.total_seconds().to_numpy()
    duracion = (lap['Time'] - lap['LapStartTime']).total_seconds()
    if (len(tiempos) < 2 or not np.isfinite(duracion) or duracion <= 0
            or tiempos[0] > 0 or tiempos[-1] < duracion):
        raise ValueError('La telemetría no cubre el inicio y el final de la vuelta.')
    return tiempos, pos['X'].to_numpy(), pos['Y'].to_numpy(), duracion


def interpolar(datos, tiempos):
    t, x, y, duracion = datos
    tiempos = np.clip(tiempos, 0, duracion)
    return np.interp(tiempos, t, x), np.interp(tiempos, t, y)


ver = posiciones_vuelta(lap_ver)
lec = posiciones_vuelta(lap_lec)
# Comparten el mismo reloj; cada piloto se detiene al completar su vuelta.
tiempos = np.linspace(0, max(ver[3], lec[3]), 450)
x_ver, y_ver = interpolar(ver, tiempos)
x_lec, y_lec = interpolar(lec, tiempos)

fig = go.Figure()
fig.add_trace(go.Scatter(x=x_ver, y=y_ver, mode='lines', name='Circuito',
                         line=dict(color='#888888', width=5), hoverinfo='skip'))
fig.add_trace(go.Scatter(
    x=circuit_info.corners['X'], y=circuit_info.corners['Y'], mode='markers+text',
    text=[f"{c['Number']}{c['Letter']}" for _, c in circuit_info.corners.iterrows()],
    textposition='middle center', textfont=dict(color='black', size=10),
    marker=dict(size=20, color='#ffe45c', line=dict(color='black', width=1)),
    name='Curvas', hoverinfo='text'))
fig.add_trace(go.Scatter(x=[x_ver[0]], y=[y_ver[0]], mode='markers',
                         name=f'Verstappen ({ver[3]:.3f} s)',
                         marker=dict(size=16, color='#1e41ff')))
fig.add_trace(go.Scatter(x=[x_lec[0]], y=[y_lec[0]], mode='markers',
                         name=f'Leclerc ({lec[3]:.3f} s)',
                         marker=dict(size=13, color='#e8002d')))

fig.frames = [go.Frame(
    name=str(i), traces=[2, 3],
    data=[go.Scatter(x=[x_ver[i]], y=[y_ver[i]]),
          go.Scatter(x=[x_lec[i]], y=[y_lec[i]])],
    layout=go.Layout(title_text=f'Verstappen vs Leclerc — Bahréin 2024<br>Tiempo: {t:.2f} s')
) for i, t in enumerate(tiempos)]

fig.update_layout(
    title='Verstappen vs Leclerc — Bahréin 2024<br>Tiempo: 0.00 s',
    height=750, xaxis=dict(visible=False),
    yaxis=dict(visible=False, scaleanchor='x', scaleratio=1),
    legend=dict(orientation='h', y=1.08),
    updatemenus=[dict(type='buttons', showactive=False, x=0, y=-0.05, buttons=[
        dict(label='▶ Reproducir', method='animate',
             args=[None, dict(frame=dict(duration=50, redraw=False),
                              transition=dict(duration=0), fromcurrent=True)]),
        dict(label='❚❚ Pausar', method='animate',
             args=[[None], dict(frame=dict(duration=0, redraw=False),
                                transition=dict(duration=0), mode='immediate')])
    ])],
    sliders=[dict(active=0, x=0.22, len=0.75, y=-0.05,
                  currentvalue=dict(prefix='Cuadro: '),
                  steps=[dict(label='', method='animate',
                              args=[[str(i)], dict(mode='immediate',
                                    frame=dict(duration=0, redraw=False),
                                    transition=dict(duration=0))])
                         for i in range(len(tiempos))])]
)
fig.show()
